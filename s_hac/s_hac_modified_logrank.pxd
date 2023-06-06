from libcpp.vector cimport vector
from libcpp cimport bool
from cpython cimport array


cdef extern from "s_hac_modified_logrank.cpp":
    vector[double] distance_computation(vector[double] surv_t, 
                                        vector[bool] surv_e,
                                        int* possible_pairs,
                                        int* neighbors,
                                        int n_pairs, 
                                        int n_neighbors)

