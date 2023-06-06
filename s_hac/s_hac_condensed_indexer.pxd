from cpython cimport array

cdef extern from "s_hac_condensed_indexer.cpp":
    int condensed_indexer_(int i, int j, int N)
