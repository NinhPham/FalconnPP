//
// Created by npha145 on 17/03/24.
//

#ifndef BF_H
#define BF_H

#include "header.h"

class bf {

    protected:

    int n_points;
    int n_features;
    int n_threads = 8;
    RowMatrixXf matrix_X;

    // function to initialize private variables
public:

    void init(int N, int d, int t, const Ref<const RowMatrixXf> & matX) {
        if (matX.rows() != N || matX.cols() != d)
            throw invalid_argument("dataset must have shape (n_points, n_features)");
        n_points = N;
        n_features = d;
        matrix_X = matX;
        n_threads = t;
    };

    MatrixXi mips_topK(const Ref<const RowMatrixXf> &, int);

    void clear() {
        matrix_X.resize(0, 0);
    };

    ~bf() { clear(); }
};


#endif //BF_H
