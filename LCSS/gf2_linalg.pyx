# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
from libc.string cimport memset, memcpy

cnp.import_array()


cdef void elim_rows_c(uint8[:, :] A, int pivot_row, int col, uint8[:, :] tracker) nogil:
    """Gaussian Elimination - optimized C-level"""
    cdef int row, k
    cdef int n_rows = A.shape[0]
    cdef int n_cols_A = A.shape[1]
    cdef int n_cols_tracker = tracker.shape[1]
    cdef uint8 pivot_val
    
    for row in range(n_rows):
        if A[row, col] != 0 and row != pivot_row:
            # XOR rows in A
            for k in range(n_cols_A):
                A[row, k] ^= A[pivot_row, k]
            
            # XOR rows in tracker
            for k in range(n_cols_tracker):
                tracker[row, k] ^= tracker[pivot_row, k]


cdef int find_nonzero_row_c(uint8[:, :] A, int pivot_row, int col) nogil:
    """Find next pivot - optimized C-level"""
    cdef int row
    cdef int n_rows = A.shape[0]
    
    for row in range(pivot_row, n_rows):
        if A[row, col] != 0:
            return row
    return -1

cpdef int find_nonzero_row(uint8[:, :] A, int pivot_row, int col):
    return find_nonzero_row_c(A, pivot_row, col)

cdef void swap_rows_c(uint8[:, :] A, int row1, int row2) nogil:
    """Swap two rows in-place"""
    cdef int k
    cdef int n_cols = A.shape[1]
    cdef uint8 temp
    
    for k in range(n_cols):
        temp = A[row1, k]
        A[row1, k] = A[row2, k]
        A[row2, k] = temp


cpdef tuple rref(uint8[:, :] A_init):
    """
    Reduced row echelon form of matrix over GF(2)
    
    Args:
        A_init: matrix (uint8)
    
    Returns:
        tuple: (rref matrix, product of elementary matrices, rank)
    """
    cdef cnp.ndarray[uint8, ndim=2] A = np.copy(A_init)
    cdef int n_rows = A.shape[0]
    cdef int n_cols = A.shape[1]
    cdef cnp.ndarray[uint8, ndim=2] tracker = np.eye(n_rows, dtype=np.uint8)
    cdef int pivot_row = 0
    cdef int col, nonzero_row
    
    cdef uint8[:, :] A_view = A
    cdef uint8[:, :] tracker_view = tracker
    
    for col in range(n_cols):
        nonzero_row = find_nonzero_row_c(A_view, pivot_row, col)
        if nonzero_row != -1:
            if nonzero_row != pivot_row:
                swap_rows_c(A_view, pivot_row, nonzero_row)
                swap_rows_c(tracker_view, pivot_row, nonzero_row)
            
            elim_rows_c(A_view, pivot_row, col, tracker_view)
            pivot_row += 1
    
    return A, tracker, pivot_row


cdef void upper(uint8[:, :] A):
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t m = A.shape[1]
    cdef Py_ssize_t i, j

    for i in range(n):
        for j in range(m):
            if i < j:
                A[i, j] = (A[i, j] ^ A[j, i]) & 1
            elif i > j:
                A[i, j] = 0
            else:
                A[i, j] = A[i, j] & 1

def gf2_nullspace(cnp.ndarray[uint8, ndim=2] A):
    """
    Compute nullspace of A over GF(2)
    
    Args:
        A: s x n matrix
    
    Returns:
        n x d matrix whose columns form basis of nullspace
    """
    cdef int s = A.shape[0]
    cdef int n = A.shape[1]
    
    R, _, rank = rref(A)
    
    cdef cnp.ndarray[uint8, ndim=2] R_arr = R
    cdef list pivots = []
    cdef int i, j, p, f
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] nz
    
    # Find pivot columns
    for i in range(rank):
        nz = np.where(R_arr[i] != 0)[0]
        if len(nz) > 0:
            pivots.append(nz[0])
    
    cdef set pivot_set = set(pivots)
    cdef list free = [j for j in range(n) if j not in pivot_set]
    
    if len(free) == 0:
        return np.zeros((n, 0), dtype=np.uint8)
    
    cdef list basis = []
    cdef cnp.ndarray[uint8, ndim=1] v
    
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for i, p in enumerate(pivots):
            v[p] = R_arr[i, f]
        basis.append(v)
    
    return np.array(basis).T & 1


def gf2_solve(cnp.ndarray[uint8, ndim=2] A, cnp.ndarray[uint8, ndim=1] b):
    """
    Solve Ax = b over GF(2)
    
    Args:
        A: s x n matrix
        b: s vector
    
    Returns:
        particular solution x or None if inconsistent
    """
    cdef int s = A.shape[0]
    cdef int n = A.shape[1]
    
    # Augment matrix [A|b]
    cdef cnp.ndarray[uint8, ndim=2] M = np.concatenate(
        [A, b.reshape(-1, 1)], axis=1
    )
    
    R, _, rank = rref(M)
    
    cdef cnp.ndarray[uint8, ndim=2] R_arr = R
    cdef int i
    cdef cnp.ndarray[uint8, ndim=1] row
    
    # Check consistency
    for i in range(s):
        if R_arr[i, n] == 1 and np.all(R_arr[i, :n] == 0):
            return None
    
    # Build particular solution
    cdef cnp.ndarray[uint8, ndim=1] x = np.zeros(n, dtype=np.uint8)
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] nz
    
    for i in range(rank):
        row = R_arr[i, :n]
        nz = np.where(row != 0)[0]
        if len(nz) > 0:
            x[nz[0]] = R_arr[i, n]
    
    return x


def gf2_block_reduce(cnp.ndarray[uint8, ndim=2] Q):
    """
    Block reduction for symplectic/quadratic forms over GF(2)
    
    Args:
        Q: symmetric matrix
    
    Returns:
        tuple: (reduced matrix, transformation matrix, block structure)
    """
    cdef cnp.ndarray[uint8, ndim=2] M = Q.copy()
    cdef int n = len(M)
    cdef list l = list(range(n))
    cdef list spaces = []
    cdef cnp.ndarray[uint8, ndim=2] P = np.eye(n, dtype=np.uint8)
    
    cdef int i, j, k
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] idx_list_np
    cdef list idx_list
    cdef uint8 beta, alpha
    
    while len(l) > 0:
        i = l[0]
        idx_list_np = np.where(M[i])[0]
        idx_list = [int(x) for x in idx_list_np if (x in l and x != i)]
        
        if len(idx_list) > 0:
            j = idx_list[0]
            for k in l:
                if k != j and k != i:
                    beta = M[min(i, k), max(i, k)]
                    alpha = M[min(j, k), max(j, k)]
                    
                    if alpha:
                        M[k] ^= M[i]
                        M[:, k] ^= M[:, i]
                        P[k] ^= P[i]
                    if beta:
                        M[k] ^= M[j]
                        M[:, k] ^= M[:, j]
                        P[k] ^= P[j]
            
            P = P & 1
            upper(M)
            M = M & 1
            l.remove(i)
            l.remove(j)
            spaces.append([i, j])
        else:
            l.remove(i)
            spaces.append([i])
    return M, P, spaces


def get_intersection(cnp.ndarray[uint8, ndim=2] V1, 
                     cnp.ndarray[uint8, ndim=2] V2,
                     cnp.ndarray[uint8, ndim=1] b1, 
                     cnp.ndarray[uint8, ndim=1] b2):
    """
    Find intersection of two affine subspaces over GF(2)
    
    Args:
        V1, V2: basis matrices
        b1, b2: offset vectors
    
    Returns:
        tuple: (particular solution, nullspace basis) or None
    """
    cdef cnp.ndarray[uint8, ndim=1] b = (b1 ^ b2) & 1
    cdef cnp.ndarray[uint8, ndim=2] A
    
    if V1.shape[1] == 0:
        A = V2
    elif V2.shape[1] == 0:
        A = V1
    else:
        A = np.hstack([V1, V2])
    
    sol = gf2_solve(A, b)
    if sol is None:
        return None
    else:
        return sol, gf2_nullspace(A)