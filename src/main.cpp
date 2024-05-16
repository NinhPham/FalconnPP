#include <iostream>
#include <ctime> // for time(0) to generate different random number

#include "Header.h"
#include "Utilities.h"
#include "InputParser.h"
#include "FalconnPP.h"
#include "BF.h"

// --numData 1183514 --n_features 200 --n_tables 10 --n_proj 256 --bucket_minSize 20, --bucket_scale 0.01
// --X "/home/npha145/Dataset/kNN/CosineKNN/Glove_X_1183514_200.txt" --n_threads 4
// --Q "/home/npha145/Dataset/kNN/CosineKNN/Glove_Q_1000_200.txt" --n_queries 1000  --qProbes 10 --n_neighbors 20 --qThreads 4

int main(int nargs, char** args) {

    IndexParam iParam;
    QueryParam qParam;

    readIndexParam(nargs, args, iParam);
    readQueryParam(nargs, args, qParam);

    // Read data
    MatrixXf MATRIX_X, MATRIX_Q;
    loadtxtDatabase(nargs, args, iParam.numPoints, iParam.numDim, MATRIX_X);
    loadtxtQuery(nargs, args, qParam.numQueries, iParam.numDim, MATRIX_Q);


    // BF
    // cout << "BF..." << endl;
    // BF bf;
    // bf.init(iParam.n_points, iParam.n_features, qParam.n_queries, qParam.n_neighbors, qParam.qThreads, MATRIX_X, MATRIX_Q);
    //
    // // Note: we should resize MatrixX and Q for saving memory
    // MATRIX_X.resize(0, 0);
    // MATRIX_Q.resize(0, 0);
    //
    // // Exact search
    // bf.mips_topK();
    // cout << "BF Finish..." << endl;
    // bf.clear();


    // 2D FalconnPP
    FalconnPP falconn(iParam.numPoints, iParam.numDim);
    falconn.Index2Layers(iParam.numTables, iParam.numProj, iParam.bucketMinSize, \
iParam.bucketScale, iParam.iProbes, iParam.numThreads);

     // 2D index
//    falconn.build2Layers(MATRIX_X);
    //    MATRIX_X.resize(0, 0); // Note: we should resize MatrixX for saving memory
//    falconn.setQueryParam(qParam.n_queries, qParam.n_neighbors, qParam.qProbes, qParam.qThreads, qParam.verbose);
//
//    for (int i = 1; i <= 10; ++i)
//    {
//        falconn.set_qProbes(1000 * i);
//        falconn.query2Layers(MATRIX_Q, qParam.topK, qParam.verbose);
//    }

    // 1D index - NeurIPS 2022
//    falconn.clear();
    falconn.build2Layers_1D(MATRIX_X);
    MATRIX_X.resize(0, 0);
    for (int i = 1; i <= 10; ++i)
    {
        falconn.set_qProbes(1000 * i);
        falconn.query2Layers_1D(MATRIX_Q, qParam.topK, qParam.verbose);
    }

    return 0;

    // MATRIX_Q.resize(0, 0); // Note: we should not resize if testing multiple qProbes
}
