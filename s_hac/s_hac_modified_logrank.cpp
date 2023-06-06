#include <vector>
#include <boost/math/distributions.hpp>
#include "fast_logrank.cpp"


std::pair<std::vector<double>, std::vector<char>> create_nn_survival_data(std::vector<double>& surv_t, std::vector<bool>& surv_e, int* neighbors, int n_neighbors, int inst_row){

    double a_t;
    bool a_e;
    std::vector<double> data_t;
    std::vector<char> data_e;
    int neighbor_idx;

    for (int n=0; n<n_neighbors; n++){
        neighbor_idx = neighbors[inst_row * n_neighbors + n];
        a_t = surv_t[neighbor_idx];
        a_e = surv_e[neighbor_idx];

        if (!isnan(a_t)){
            data_t.push_back(a_t);
            data_e.push_back(a_e);
        }
    }
    return std::make_pair(data_t, data_e);
}


std::vector<double> distance_computation(std::vector<double>& surv_t, std::vector<bool>& surv_e, int* possible_pairs, int* neighbors, int n_pairs, int n_neighbors){
    
    std::vector<double> distances;
    std::pair<std::vector<double>, std::vector<char>> group_a_te;
    std::pair<std::vector<double>, std::vector<char>> group_b_te;

    std::vector<double> group_a_t;
    std::vector<double> group_b_t;
    std::vector<char> group_a_e;
    std::vector<char> group_b_e;

    double distance;
    
    int inst_a = -1; 
    int inst_b;
     
    for (int i=0; i<n_pairs*2; i=i+2){
        
        // The first index often repeats n_neighbors times in a row. Only regather survival data, if the index changes.
        if (inst_a != possible_pairs[i]){
            inst_a = possible_pairs[i];
            group_a_te = create_nn_survival_data(surv_t, surv_e, neighbors, n_neighbors, inst_a);
            group_a_t = group_a_te.first;
            group_a_e = group_a_te.second;
        }
        
        inst_b = possible_pairs[i+1];
        group_b_te = create_nn_survival_data(surv_t, surv_e, neighbors, n_neighbors, inst_b);
        group_b_t = group_b_te.first;
        group_b_e = group_b_te.second;

        distance = logrank_instance(group_a_t, group_a_e, group_b_t, group_b_e, true);
        distances.push_back(distance);
    }
    return distances;
}
