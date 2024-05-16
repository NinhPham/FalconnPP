//
// Created by npha145 on 17/03/24.
//

#ifndef FALCONNCEOS_BF_H
#define FALCONNCEOS_BF_H

#include "Header.h"

class BF {

    protected:

    int numPoints;
    int numDim;

    int numQueries;
    int topK = 1;

    int qThreads = -1;

    MatrixXf matrix_Q;
    MatrixXf matrix_X;

    // function to initialize private variables
public:

    void init(int N, int d, int Q, int k, int t, const MatrixXf& matX, const MatrixXf& matQ) {
        numPoints = N;
        numDim = d;
        matrix_X = matX;

        matrix_Q = matQ;
        numQueries = Q;
        topK = k;
        qThreads = t;
    };

    MatrixXi mips_topK();

    void clear() {
        matrix_Q.resize(0, 0);
        matrix_X.resize(0, 0);
    };
};


#endif //FALCONNCEOS_BF_H
