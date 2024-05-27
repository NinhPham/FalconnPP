//
// Created by npha145 on 17/03/24.
//

#ifndef BF_H
#define BF_H

#include "Header.h"

class BF {

    protected:

    int n_points;
    int n_features;
    int n_threads = 8;
    MatrixXf matrix_X;

    // function to initialize private variables
public:

    void init(int N, int d, int t, const Ref<const MatrixXf> & matX) {
        n_points = N;
        n_features = d;
        matrix_X = matX;
        n_threads = t;
    };

    MatrixXi mips_topK(const Ref<const MatrixXf> &, int);

    void clear() {
        matrix_X.resize(0, 0);
    };

    ~BF() { clear(); }
};


#endif //BF_H
