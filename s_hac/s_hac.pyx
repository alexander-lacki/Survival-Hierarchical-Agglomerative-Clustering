# distutils: language=c++
from s_hac_modified_logrank cimport distance_computation
from s_hac_condensed_indexer cimport condensed_indexer_
from libcpp cimport bool
from libcpp.vector cimport vector

# Python Imports
import numpy as np
from multiprocessing import Pool
from itertools import product
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from fastlogranktest import logrank_test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def distance_computation_cy(vector[double] surv_t, 
                            vector[bool] surv_e, 
                            int[:, ::1] possible_pairs,
                            int[:, ::1] neighbors, 
                            int n_pairs, 
                            int n_neighbors):
    return distance_computation(surv_t, surv_e, &possible_pairs[0, 0], &neighbors[0, 0],  n_pairs, n_neighbors)


def distance_computation_cy_(surv_t, surv_e, possible_pairs, neighbors):
    n_pairs = possible_pairs.shape[0]
    n_neighbors = neighbors.shape[1]
    
    return distance_computation_cy(surv_t, surv_e, possible_pairs, neighbors, n_pairs, n_neighbors)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def condensed_indexer_cy(int i, int j, int N):
    return condensed_indexer_(i, j, N)


def condensed_indexer(i, j, N):
    return condensed_indexer_cy(i, j, N)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class Node:
    def __init__(self, data):
        self.item = data
        self.next = None
        self.prev = None
        
    def remove(self, active_start):
        prev = self.prev
        nxt  = self.next
        
        if prev is None:
            nxt.prev = None
            return nxt
            
        elif nxt is None:
            prev.next = None
            return active_start
        else:
            prev.next, nxt.prev = nxt, prev
            return active_start
        
    def length(self):
        n = self
        
        while(1):
            if n.prev is not None:
                n = n.prev
            else:
                break
        
        ctr = 0
        
        while(1):
            if n.next is not None:
                n = n.next
                ctr += 1
            else:
                break
        return ctr
        

class doublyLinkedList:
    def __init__(self):
        self.start_node = None
        
    def insert(self, data):
        
        # Check if the list is empty
        if self.start_node is None:
            new_node = Node(data)
            self.start_node = new_node
            return
        
        n = self.start_node
        # Iterate till the next reaches NULL
        
        while n.next is not None:
            n = n.next
        
        new_node = Node(data)
        n.next = new_node
        new_node.prev = n


class SurvivalHierarchicalAgglomerativeClustering:
    def __init__(self, n_neighbors, alpha, min_cluster_size, logrank_p_threshold, processes=8):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.min_cluster_size = min_cluster_size
        self.logrank_p_threshold = logrank_p_threshold
        self.dist_mx = None
        self.processes = processes
        
        
    def logrank_p(self, treat_a_list, treat_b_list):
        
        min_p = [1.0]
        
        for treat_a, treat_b in zip(treat_a_list, treat_b_list):
            treat_a = np.array(treat_a)
            treat_b = np.array(treat_b)
            
            treat_a = treat_a[~np.isnan(treat_a[:, 0]), :]
            treat_b = treat_b[~np.isnan(treat_b[:, 0]), :]

            p_val = logrank_test(treat_a[:, 0], 
                                 treat_b[:, 0], 
                                 treat_a[:, 1], 
                                 treat_b[:, 1])
            
            min_p.append(p_val)
        
        p_val = np.nanmin(min_p)
        
        return p_val
    
    
    def pairwise_survival_generator(self, lifetimes, neighbors, pp):
        return distance_computation_cy_(lifetimes[:, 0], 
                                        lifetimes[:, 1].astype(np.int8), 
                                        np.ascontiguousarray(np.array(pp).T).astype(np.int32), 
                                        neighbors.astype(np.int32))
    

    def compute_surv_distance_mx(self, lifetimes_array, neighbors, possible_pairs):
        p = Pool(processes=self.processes)
        surv_dists = p.starmap(self.pairwise_survival_generator, ((lifetimes_array, 
                                                                  neighbors, 
                                                                  pp) for pp in possible_pairs))
        p.close()
        p.join()
        
        surv_dist = np.zeros((lifetimes_array.shape[0], lifetimes_array.shape[0]))
        
        for surv_dists_, pps in zip(surv_dists, possible_pairs):
            for i in range(len(surv_dists_)):
                surv_dist[pps[0, i], pps[1, i]] = surv_dists_[i]

        # Fix Symmetry
        surv_dist = np.maximum(surv_dist, surv_dist.T)
        surv_dist[surv_dist==np.nan] = np.mean(surv_dist[surv_dist!=np.nan])

        return surv_dist

        
    def compute_ss_distance_matrix(self, X, lifetimes_list, precomputed_distMx=None):
        # lifetimes_list = np.array(lifetimes_list[0])
        
        if precomputed_distMx is None:
            knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
            knn.fit(X)
        else:
            knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='precomputed')
            knn.fit(precomputed_distMx)
            
        neighbors = knn.kneighbors(return_distance=False)
        
        possible_pairs = np.triu_indices(X.shape[0], 1)
        possible_pairs = np.vstack(possible_pairs)
        possible_pairs = np.array_split(possible_pairs, self.processes, axis=1)
        
        surv_dist_matrices = []
        
        for lifetimes_array in lifetimes_list:
            surv_dist_mx = self.compute_surv_distance_mx(lifetimes_array, neighbors, possible_pairs)
            surv_dist_matrices.append(surv_dist_mx)
        
        self.surv_dist = np.max(surv_dist_matrices, axis=0) if len(surv_dist_matrices) > 1 else surv_dist_matrices[0]
        self.cov_dist = distance_matrix(X, X) if precomputed_distMx is None else precomputed_distMx
        
        # Standardize Distance Matrices
        self.cov_dist /= np.mean(self.cov_dist)
        self.surv_dist /= np.mean(self.surv_dist)
            
        # Combine
        comb_dmx = (1-self.alpha) * self.cov_dist + self.alpha * self.surv_dist
        
        return comb_dmx
    
    
    def cluster(self, dist_mx, lifetimes_list, min_cluster_size, X):
        n_patients = dist_mx.shape[0]
        curr_clusts = {p_id: [p_id] for p_id in range(n_patients)}
        prev_clusts = {}
        min_n_cluster = min_cluster_size
        
        sq_dist_mx = squareform(dist_mx)
        linkage_matrix = linkage(sq_dist_mx, "average")

        for i in range(linkage_matrix.shape[0]):
            clust_id_1 = linkage_matrix[i, 0]
            clust_id_2 = linkage_matrix[i, 1]
            n_on_agglomeration = linkage_matrix[i, -1]

            if clust_id_1 not in curr_clusts or clust_id_2 not in curr_clusts:
                # No agglomeration possible. Agglomeration for one of the candidates was previously rejected
                # Consider agglomerating on children if minimum cluster size unreached
                if clust_id_1 in curr_clusts or clust_id_2 in curr_clusts:
                    present_clust_id = clust_id_1 if clust_id_1 in curr_clusts else clust_id_2
                    missing_clust_id = clust_id_1 if clust_id_1 not in curr_clusts else clust_id_2

                    if len(curr_clusts[present_clust_id]) < min_n_cluster:
                        old_agglomeration_row = int(missing_clust_id - n_patients)
                        sub_clusts = linkage_matrix[old_agglomeration_row, :2]

                        cand_clusts = set(sub_clusts)

                        while True:
                            if any([cand_clust in curr_clusts for cand_clust in cand_clusts]):
                                break
                            else:
                                pass

                            for cand_clust in cand_clusts.copy():
                                if not cand_clust in curr_clusts:
                                    if int(cand_clust - n_patients) >= 0:
                                        old_agglomeration_row = int(cand_clust - n_patients)
                                        sub_clusts = list(linkage_matrix[old_agglomeration_row, :2])
                                    else:
                                        sub_clusts = [cand_clust]

                                    # print("Cluster ", old_a)
                                    cand_clusts.update(sub_clusts)
                                    cand_clusts.remove(cand_clust)

                        cand_clusts = [cand_clust for cand_clust in list(cand_clusts) if cand_clust in curr_clusts]
                        cand_idcs = [[p_id for p_id in curr_clusts[cand_clust]] for cand_clust in cand_clusts]

                        idcs_present_clust = [p_id for p_id in curr_clusts[present_clust_id]]
                        distances = [np.mean([[dist_mx[x, y] for x in cand_idcs_] for y in idcs_present_clust]) for cand_idcs_ in cand_idcs]

                        idx = np.argmin(distances)

                        curr_clusts[cand_clusts[idx]] += curr_clusts[present_clust_id]
                        curr_clusts.pop(present_clust_id)
                        # print("Agglomerated to child")

            elif len(curr_clusts[clust_id_1]) <= min_n_cluster or len(curr_clusts[clust_id_2]) <= min_n_cluster:
                # Minimum cluster size not reached -> Agglomerate
                # print("Agglomerating for minimum distance: ", clust_id_1, clust_id_2)
                curr_clusts[i + n_patients] = curr_clusts[clust_id_1] + curr_clusts[clust_id_2]

                prev_clusts[clust_id_1] = curr_clusts[clust_id_1]
                prev_clusts[clust_id_2] = curr_clusts[clust_id_2]

                curr_clusts.pop(clust_id_1)
                curr_clusts.pop(clust_id_2)

            else:
                # Minimum size reached, check for LogRank p-val
                p_ids_1 = curr_clusts[clust_id_1]
                p_ids_2 = curr_clusts[clust_id_2]

                treats_1 = [[lifetimes[id_] for id_ in p_ids_1] for lifetimes in lifetimes_list]
                treats_2 = [[lifetimes[id_] for id_ in p_ids_2] for lifetimes in lifetimes_list]
                
                if self.logrank_p(treats_1, treats_2) < self.logrank_p_threshold:
                    # Do not agglomerate
                    continue
                else:
                    # Agglomerate
                    curr_clusts[i + n_patients] = curr_clusts[clust_id_1] + curr_clusts[clust_id_2]

                    prev_clusts[clust_id_1] = curr_clusts[clust_id_1]
                    prev_clusts[clust_id_2] = curr_clusts[clust_id_2]

                    curr_clusts.pop(clust_id_1)
                    curr_clusts.pop(clust_id_2)

        p_id_c_ass = {}
        p_ids = list(range(n_patients))
        clust_assignments = np.zeros((n_patients))

        for clust, cc in enumerate(curr_clusts):
            for p_id in curr_clusts[cc]:
                idx = p_ids.index(p_id)
                clust_assignments[idx] = clust

        clust_assignments = clust_assignments.astype(int)

        return clust_assignments, linkage_matrix
    
    
    def compute_distance_between_clusters(self, idcs_1, idcs_2 , dist_mx):
        a, b = np.meshgrid(idcs_1, idcs_2, copy=False)
        return np.mean(dist_mx[a, b])
    
    
    def find_NN(self, n, idx_0, D, C, N):
        min_d = np.inf
        idx = None
        
        while(n is not None):
            if idx_0 == n.item:
                n = n.next
                continue
                
            mx_idx = condensed_indexer(idx_0, n.item, N)
            
            if not C[mx_idx]:
                
                d = D[mx_idx]

                if d < min_d:
                    min_d = d
                    idx = n.item
                
            n = n.next
        return idx
    
    
    def postprocess_clusters(self, X, lifetimes_list, cluster_assignments, dist_mx):
        c_ids, cnts = np.unique(cluster_assignments, return_counts=True)
        c_ids = list(c_ids)
        cnts = list(cnts)

        clust_dict = {key: value for key, value in [(c_id, np.where(cluster_assignments==c_id)[0]) for c_id in c_ids]}
        
        next_key = max(list(clust_dict.keys())) + 1

        # Dissolve microclusters into singletons
        for c_id in clust_dict.copy():
            if len(clust_dict[c_id]) < self.min_cluster_size:
                for idx in clust_dict[c_id]:
                    clust_dict[next_key] = [idx]
                    next_key += 1

                clust_dict.pop(c_id)
            else:
                clust_dict[c_id] = list(clust_dict[c_id])
        
        # Reset cluster keys to range. This range will be maintained until the end
        clust_dict = {i: clust_dict[key] for i, key in enumerate(clust_dict.keys())}
        
        # Initialize distance dict
        poss_c_pairs = product(clust_dict.keys(), repeat=2)
        poss_c_pairs = [(c1, c2) for c1, c2 in poss_c_pairs if c1<c2]
        
        # Compute distances and link constraints between clusters
        N = len(clust_dict)
        distances = np.zeros(int(N*(N-1)/2), dtype=float)
        link_constraint = np.zeros(int(N*(N-1)/2),dtype=np.int8)
        
        tmp_dist_mx = memoryview(self.dist_mx)
        
        for c_pair in poss_c_pairs:
            idcs_1 = clust_dict[c_pair[0]]
            idcs_2 = clust_dict[c_pair[1]]
            
            d_idx = condensed_indexer(c_pair[0], c_pair[1], N)
            
            if len(idcs_1) == 1 and len(idcs_2) == 1:
                link_constraint[d_idx] = True
                distances[d_idx] = tmp_dist_mx[idcs_1[0], idcs_2[0]]
            else:
                distances[d_idx] = self.compute_distance_between_clusters(idcs_1, idcs_2, dist_mx)
            
        del poss_c_pairs
        del tmp_dist_mx
        
        D = memoryview(distances)
        C = memoryview(link_constraint)
        
        D_i = lambda x,y : D[condensed_indexer(x, y, N)]
        C_i = lambda x,y : C[condensed_indexer(x, y, N)]
        
        
        NN_chain = []
        active_nodes = doublyLinkedList()
        [active_nodes.insert(x) for x in range(N)]
        
        # print("Running NN Chain. Clusters:", N)
        
        n_outer = active_nodes.start_node
            
        while(1):
            NN_chain = []
            
            if n_outer is None:
                break
            
            # Find candidate pair
            idx_0 = n_outer.item
            NN_chain.append(idx_0)
            
            while(1):
                n = active_nodes.start_node
                idx_0 = self.find_NN(n, idx_0, D, C, N)
                
                if idx_0 in NN_chain or idx_0 is None:
                    break
                
                NN_chain.append(idx_0)
            
            
            if idx_0 is None:
                # No unconstrained for n_outer
                n_outer = n_outer.next
                continue
                    
            # Found Pair
            c1, c2 = NN_chain[-2:]
            
            p_ids_1 = clust_dict[c1]
            p_ids_2 = clust_dict[c2]

            if len(p_ids_1) > 1 and len(p_ids_2) > 1:
                treats_1 = [lifetimes[p_ids_1] for lifetimes in lifetimes_list]
                treats_2 = [lifetimes[p_ids_2] for lifetimes in lifetimes_list]

                p_val = self.logrank_p(treats_1, treats_2)
            else:
                p_val = 1.0

            if p_val > self.logrank_p_threshold:
                # Merge clusters
                # Recompute_distance
                
                s1 = len(p_ids_1)
                s2 = len(p_ids_2)
                
                n = active_nodes.start_node
                
                while(n is not None):
                    if n.item == c2:
                        active_nodes.start_node = n.remove(active_nodes.start_node)
                        n = n.next
                        continue
                        
                    if n.item == c1:
                        n = n.next
                        continue
                    
                    c1_mx_idx = condensed_indexer(n.item, c1, N)
                    
                    d_1 = D[c1_mx_idx] * s1
                    d_2 = D_i(n.item, c2) * s2
                    
                    d = (d_1 + d_2) / (s1 + s2)
                    
                    D[c1_mx_idx] = d
                    C[condensed_indexer(n.item, c1, N)] = False
                    
                    n = n.next
                    
                
                clust_dict[c1] = clust_dict[c1] + clust_dict[c2]
                del clust_dict[c2]
                
            else:
                # Rejecting Merge
                # Set cannot-link contraints to non-singular clusters
                C[condensed_indexer(c1, c2, N)] = True
                
                n = active_nodes.start_node
                
                while(n is not None):
                    if n.item == c1 or n.item == c2:
                        n = n.next
                        continue
                    if len(clust_dict[n.item]) > 1:
                        C[condensed_indexer(n.item, c1, N)] = True
                        C[condensed_indexer(n.item, c2, N)] = True
                    n = n.next
        
        # Create cluster_assigment vector
        cluster_assigment = np.zeros(sum(map(len, clust_dict.values())))

        for i, c_id in enumerate(clust_dict):
            cluster_assigment[clust_dict[c_id]] = i

        return cluster_assigment
    
    
    def fit_predict(self, X, lifetimes, precomputed_distance=None):
        
        if self.dist_mx is None:
            self.dist_mx = self.compute_ss_distance_matrix(X, lifetimes, precomputed_distance)
        
        init_clust_assignments, linkage_matrix = self.cluster(self.dist_mx, lifetimes, 8, X)
        # print("No. clusts preclust:", np.unique(init_clust_assignments))
        # print("postclust", flush=True)
        clust_assignments_2 = self.postprocess_clusters(X, lifetimes, init_clust_assignments, self.dist_mx)
        
        return clust_assignments_2.astype(int)

