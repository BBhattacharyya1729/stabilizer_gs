cimport numpy as cnp
ctypedef cnp.uint8_t uint8

cpdef int find_nonzero_row(uint8[:, :] A, int pivot_row, int col)
cpdef tuple rref(uint8[:, :] A_init)
cdef void upper(uint8[:, :] A)
