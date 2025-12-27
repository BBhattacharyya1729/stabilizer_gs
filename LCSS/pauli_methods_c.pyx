# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython

from gf2_linalg import (
    gf2_nullspace,
    gf2_solve
)
from gf2_linalg cimport find_nonzero_row, rref


cnp.import_array()

ctypedef cnp.uint8_t uint8

# -------------------------------------------------------------------
# Pauli lookup tables (compile-time constants)
# -------------------------------------------------------------------

cdef uint8 PAULI_MAP[4][2]
PAULI_MAP[0][0] = 0; PAULI_MAP[0][1] = 0   # '_'
PAULI_MAP[1][0] = 1; PAULI_MAP[1][1] = 0   # 'X'
PAULI_MAP[2][0] = 1; PAULI_MAP[2][1] = 1   # 'Y'
PAULI_MAP[3][0] = 0; PAULI_MAP[3][1] = 1   # 'Z'

cdef char PAULI_INV[2][2]
PAULI_INV[0][0] = ord('_')
PAULI_INV[1][0] = ord('X')
PAULI_INV[1][1] = ord('Y')
PAULI_INV[0][1] = ord('Z')

cdef inline int pauli_index(char c) nogil:
    if c == 'X': return 1
    if c == 'Y': return 2
    if c == 'Z': return 3
    return 0   # '_'


# -------------------------------------------------------------------
# Pauli <-> binary
# -------------------------------------------------------------------

def pauli_to_bin(str s):
    """
    Convert Pauli string to binary tableau row.
    """
    cdef Py_ssize_t n = len(s) - 1
    cdef cnp.ndarray[uint8, ndim=1] out = np.zeros(2*n + 1, dtype=np.uint8)

    cdef int i, idx
    for i in range(n):
        idx = pauli_index(s[i+1])
        out[i] = PAULI_MAP[idx][0]
        out[i+n] = PAULI_MAP[idx][1]

    out[2*n] = 0 if s[0] == '+' else 1
    return out


def bin_to_pauli(cnp.ndarray[uint8, ndim=1] arr):
    """
    Convert binary tableau row back to Pauli string.
    """
    cdef int n = (arr.shape[0] - 1) // 2
    cdef list chars = []
    cdef int i

    for i in range(n):
        chars.append(chr(PAULI_INV[arr[i]][arr[i+n]]))

    return ('+' if arr[-1] == 0 else '-') + ''.join(chars)


# -------------------------------------------------------------------
# Group / symplectic utilities
# -------------------------------------------------------------------

def get_forbidden(list l, list paulis):
    cdef cnp.ndarray[uint8, ndim=2] current_v = np.array(
        [pauli_to_bin(p)[:-1] for p in l], dtype=np.uint8
    )
    cdef cnp.ndarray[uint8, ndim=2] S = np.array(
        [pauli_to_bin('+' + p)[:-1] for p in paulis], dtype=np.uint8
    )

    cdef list forbidden = []
    cdef int i

    for i in range(S.shape[0]):
        if gf2_solve(current_v.T, S[i]) is None:
            forbidden.append(S[i])

    return np.array(forbidden, dtype=np.uint8)


def symplectic_complement(list l):
    cdef cnp.ndarray[uint8, ndim=2] V = np.array(
        [pauli_to_bin(p)[:-1] for p in l], dtype=np.uint8
    )

    cdef int n = V.shape[1] // 2

    cdef cnp.ndarray[uint8, ndim=2] P = np.block([
        [np.zeros((n,n), dtype=np.uint8), np.eye(n, dtype=np.uint8)],
        [np.eye(n, dtype=np.uint8), np.zeros((n,n), dtype=np.uint8)]
    ])

    cdef cnp.ndarray[uint8, ndim=2] W = gf2_nullspace(V @ P)

    cdef cnp.ndarray[uint8, ndim=2] A = np.hstack([V.T, W])
    cdef cnp.ndarray[uint8, ndim=2] A_red = rref(A)[0][:, V.shape[0]:]

    cdef list pivots = []
    cdef int pivot_row = V.shape[0]
    cdef int col, row

    for col in range(A_red.shape[1]):
        row = find_nonzero_row(A_red, pivot_row, col)
        if row >= 0:
            pivots.append(col)
            pivot_row = row + 1

    return W[:, pivots], V.T


def remove_forbidden(cnp.ndarray[uint8, ndim=2] V,
                     cnp.ndarray[uint8, ndim=2] W,
                     cnp.ndarray[uint8, ndim=1] f):

    cdef cnp.ndarray sol = gf2_solve(np.hstack([V, W]), f)
    if sol is None:
        return W

    cdef cnp.ndarray[uint8, ndim=1] w = W @ sol[V.shape[1]:]

    cdef cnp.ndarray[uint8, ndim=2] A = np.vstack([w, W.T]).T
    cdef cnp.ndarray[uint8, ndim=2] A_red = rref(A)[0][:, 1:]

    cdef list pivots = []
    cdef int pivot_row = 1
    cdef int col, row

    for col in range(A_red.shape[1]):
        row = find_nonzero_row(A_red, pivot_row, col)
        if row >= 0:
            pivots.append(col)
            pivot_row = row + 1

    if W.shape[1] > 1:
        return W[:, pivots]
    else:
        return None


def extend_group(list l, cnp.ndarray forbidden):
    cdef cnp.ndarray W, V
    W, V = symplectic_complement(l)

    cdef int i
    for i in range(forbidden.shape[0]):
        W = remove_forbidden(V, W, forbidden[i])
        if W is None:
            return None

    return bin_to_pauli(
        np.concatenate([W[:,0], np.zeros(1, dtype=np.uint8)])
    )


def full_extend_group(list l, list paulis):
    cdef int n = len(paulis[0])
    cdef cnp.ndarray forbidden = get_forbidden(l, paulis)

    while len(l) < n:
        new_op = extend_group(l, forbidden)
        if new_op is None:
            return None
        l.append(new_op)

    return l


# -------------------------------------------------------------------
# Phase-aware Gaussian elimination
# -------------------------------------------------------------------

cdef void phase_add_rows(uint8[:] v1, uint8[:] v2) noexcept nogil:
    cdef int L = v1.shape[0]
    cdef int n = (L - 1) // 2
    cdef int i

    cdef int t0 = 0   # q2 @ p1
    cdef int t1 = 0   # q1 @ p1 + q2 @ p2 + 3 * (...)
    cdef int c

    # t0 = q2 @ p1
    for i in range(n):
        t0 += v2[i] * v1[n + i]

    # q1 @ p1 + q2 @ p2
    for i in range(n):
        t1 += v1[i] * v1[n + i]
        t1 += v2[i] * v2[n + i]

    # 3 * ((q1 + q2)%2 @ (p1 + p2)%2)
    for i in range(n):
        t1 += 3 * ((v1[i] + v2[i]) & 1) * ((v1[n+i] + v2[n+i]) & 1)

    c = t0 + (t1 // 2)

    # row addition mod 2
    for i in range(L):
        v2[i] ^= v1[i]

    # phase update
    v2[L - 1] ^= (c & 1)



def phase_elim_rows(cnp.ndarray[uint8, ndim=2] A,
                    int pivot_row,
                    int col,
                    cnp.ndarray[uint8, ndim=2] tracker):

    cdef uint8[:, :] A_view = A
    cdef uint8[:, :] T_view = tracker
    cdef int row
    cdef int k, ncols = T_view.shape[1]
    for row in range(A.shape[0]):
        if A_view[row, col] and row != pivot_row:
            phase_add_rows(A_view[pivot_row], A_view[row])
            for k in range(ncols):
                T_view[row, k] ^= T_view[pivot_row, k]



def phase_rref(cnp.ndarray[uint8, ndim=2] A_init):
    cdef cnp.ndarray[uint8, ndim=2] A = np.copy(A_init)
    cdef cnp.ndarray[uint8, ndim=2] tracker = np.eye(A.shape[0], dtype=np.uint8)

    cdef int pivot_row = 0
    cdef int col, row

    for col in range(A.shape[1]):
        row = find_nonzero_row(A, pivot_row, col)
        if row >= 0:
            A[[pivot_row, row]] = A[[row, pivot_row]]
            tracker[[pivot_row, row]] = tracker[[row, pivot_row]]
            phase_elim_rows(A, pivot_row, col, tracker)
            pivot_row += 1

    return A, tracker, pivot_row


def canonical(list group):
    cdef int n = len(group)
    cdef cnp.ndarray[uint8, ndim=2] A = np.array(
        [pauli_to_bin(p) for p in group], dtype=np.uint8
    )

    A, _, _ = phase_rref(A)

    cdef int i
    for i in range(A.shape[0]):
        A[i, -1] ^= ((A[i,:n] @ A[i,n:2*n]) % 4) // 2

    return A & 1
