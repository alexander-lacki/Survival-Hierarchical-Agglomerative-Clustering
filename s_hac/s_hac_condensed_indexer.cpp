int condensed_indexer_(int i, int j, int N){
    if (i < j){
        return N * i + j - ((i + 2) * (i + 1)) / 2;
    }else{
        return N * j + i - ((j + 2) * (j + 1)) / 2;
    }
}
