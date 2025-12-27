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
    gf2_solve,
    gf2_block_reduce,
    get_intersection,
)
from gf2_linalg cimport upper

from pauli_methods_c import (
    canonical,
    pauli_to_bin
)

cnp.import_array()
ctypedef cnp.uint8_t uint8

cdef class triple:
    cdef public cnp.ndarray l
    cdef public cnp.ndarray Q
    cdef public cnp.ndarray V
    cdef public cnp.ndarray z0
    cdef public int n
    cdef public int k

    def __init__(self, l, Q, V, z0):
        self.l  = np.asarray(l)
        self.Q  = np.asarray(Q)
        self.V  = np.asarray(V)
        self.z0 = np.asarray(z0)

        self.n = self.z0.shape[0]
        self.k = self.Q.shape[0]

    @staticmethod
    def get_triple(group):
        A = canonical(group)
        n = A.shape[0]

        q = []
        p = []
        rho = []
        c = []
        gamma = []

        for i in range(n):
            if np.any(A[i, :n] != 0):
                q.append(A[i, :n])
                p.append(A[i, n:2*n])
                c.append(A[i, -1])
            else:
                rho.append(A[i, n:2*n])
                gamma.append(A[i, -1])

        q = np.array(q, dtype=np.uint8)
        p = np.array(p, dtype=np.uint8)
        rho = np.array(rho, dtype=np.uint8)
        gamma = np.array(gamma, dtype=np.uint8)

        z0 = np.zeros(n, dtype=np.uint8)
        if gamma.shape[0] > 0:
            z0 = gf2_solve(rho, gamma) % 2

        k = q.shape[0]
        l = np.array([(x @ y) for (x, y) in zip(p, q)], dtype=np.uint8) % 2

        Q = np.zeros((k, k), dtype=np.uint8)

        for i in range(k):
            Q[i, i] = (c[i] + (p[i] @ z0)) % 2
            for j in range(i + 1, k):
                Q[i, j] = ((p[j] @ q[i]) + (p[i] @ q[i]) * (p[j] @ q[j])) % 2

        Q &= 1
        return triple(l, Q, q, z0)

    @staticmethod
    def __inner_terms__(triple triple1, triple triple2):
        overlap = get_intersection(
            triple1.V.T,
            triple2.V.T,
            triple1.z0,
            triple2.z0
        )

        if overlap is None:
            return 0, np.zeros(0, dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8)

        b, M = overlap

        N = (1 / np.sqrt(2 ** triple1.k)) * (1 / np.sqrt(2 ** triple2.k))

        phase = (
            (-1 + 0j) ** (b[:triple1.k] @ triple1.Q @ b[:triple1.k])
            * (-1 + 0j) ** (b[triple1.k:] @ triple2.Q @ b[triple1.k:])
            * (-1j) ** ((triple1.l @ b[:triple1.k]) % 2)
            * (1j) ** ((triple2.l @ b[triple1.k:]) % 2)
        )

        if M.shape[1] == 0:
            return phase * N, np.zeros(0, dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8)

        P1 = M[:triple1.k]
        P2 = M[triple1.k:]

        l1_new = P1.T @ triple1.l
        l2_new = P2.T @ triple2.l
        l_new = (l1_new + l2_new) % 2

        Q_new = (
            P1.T @ triple1.Q @ P1
            + P2.T @ triple2.Q @ P2
            + np.outer(l1_new, l2_new)
        ) % 2

        Ql = (
            P1.T @ (triple1.Q + triple1.Q.T) @ b[:triple1.k]
            + P2.T @ (triple2.Q + triple2.Q.T) @ b[triple1.k:]
            + P1.T @ np.outer(triple1.l, triple1.l) @ b[:triple1.k]
            + P2.T @ np.outer(triple2.l, triple2.l) @ b[triple1.k:]
            + l1_new
        ) % 2

        Ql = (Ql + np.diag(Q_new)) % 2
        np.fill_diagonal(Q_new, Ql)

        return phase * N, l_new, Q_new

    @staticmethod
    def __block_sum__(Qb, blocks):
        prod = 1
        for b in blocks:
            if len(b) == 1:
                if Qb[b, b] == 0:
                    prod *= 2
                else:
                    return 0
            else:
                p = (
                    1
                    + (-1 + 0j) ** Qb[b[0], b[0]]
                    + (-1 + 0j) ** Qb[b[1], b[1]]
                    + (-1 + 0j) ** (
                        Qb[b[0], b[0]]
                        + Qb[b[0], b[1]]
                        + Qb[b[1], b[0]]
                        + Qb[b[1], b[1]]
                    )
                )
                if p == 0:
                    return 0
                prod *= p
        return prod

    @staticmethod
    def __twisted_block_sum__(Q, l):
        Qb, B, blocks = gf2_block_reduce(Q)
        lb = B @ l
        Qlb = (Qb + np.diag(lb)) % 2

        quad_term = triple.__block_sum__(Qb, blocks)
        lin_term = triple.__block_sum__(Qlb, blocks)

        return (quad_term + lin_term) / 2 + 1j * (quad_term - lin_term) / 2

    @staticmethod
    def inner(triple triple1, triple triple2):
        p, l, Q = triple.__inner_terms__(triple1, triple2)
        upper(Q)
        return p * triple.__twisted_block_sum__(Q, l)

    def apply_pauli(self, P):
        v = pauli_to_bin("+" + P)
        q = v[:len(P)]
        p = v[len(P):2 * len(P)]

        Q_new = self.Q + np.diag(self.V @ p)
        z0_new = self.z0 + q

        phase = (1j) ** (q @ p) * (-1 + 0j) ** (p @ self.z0)

        return phase, triple(self.l, Q_new, self.V, z0_new)