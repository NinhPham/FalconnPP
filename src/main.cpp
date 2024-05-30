#include <iostream>
#include <ctime> // for time(0) to generate different random number

#include "Header.h"
#include "Utilities.h"
#include "FalconnPP.h"
#include "BF.h"

// --numData 1183514 --n_features 200 --n_tables 10 --n_proj 256 --bucket_minSize 20, --bucket_scale 0.01
// --X "/home/npha145/Dataset/kNN/CosineKNN/Glove_X_1183514_200.txt" --n_threads 4
// --Q "/home/npha145/Dataset/kNN/CosineKNN/Glove_Q_1000_200.txt" --n_queries 1000  --qProbes 10 --n_neighbors 20 --n_threads 4

int main(int nargs, char** args) {

    IndexParam iParam;
    QueryParam qParam;

    readIndexParam(nargs, args, iParam);
    readQueryParam(nargs, args, qParam);

    // Read data
    MatrixXf MATRIX_X, MATRIX_Q;

    // Read dataset
    string dataset = "";
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--X") == 0) {
            dataset = args[i + 1]; // convert char* to string
            break;
        }
    }
    if (dataset == "") {
        cerr << "Error: Data file does not exist !" << endl;
        exit(1);
    }
    else
        loadtxtData(dataset, iParam.n_points, iParam.n_features, MATRIX_X);

    // Read query set
    dataset = "";
    for (int i = 1; i < nargs; i++) {
        if (strcmp(args[i], "--Q") == 0) {
            dataset = args[i + 1]; // convert char* to string
            break;
        }
    }
    if (dataset == "") {
        cerr << "Error: Query file does not exist !" << endl;
        exit(1);
    }
    else
        loadtxtData(dataset, qParam.n_queries, iParam.n_features, MATRIX_Q);

//     BF bf;
//     bf.init(iParam.n_points, iParam.n_features, iParam.n_threads, MATRIX_X);
//
//     // Note: we should resize MatrixX and Q for saving memory
//     MATRIX_X.resize(0, 0);
//     MATRIX_Q.resize(0, 0);
//
//     // Exact search
//     bf.mips_topK(MATRIX_Q, qParam.n_neighbors);
//     cout << "BF Finish..." << endl;
//     bf.clear();


    // 2D FalconnPP
    FalconnPP falconn(iParam.n_points, iParam.n_features);
    falconn.Index2Layers(iParam.n_tables, iParam.n_proj, iParam.bucket_minSize, \
    iParam.bucket_scale, iParam.iProbes, iParam.n_threads, iParam.seed);

    // 1D index - NeurIPS 2022
//    falconn.clear();
    falconn.build2Layers_1D(MATRIX_X);
    MATRIX_X.resize(0, 0); // if not use MATRIX_X later,  need to resize to save memory
    for (int i = 1; i <= 10; ++i)
    {
        falconn.set_qProbes(1000 * i);
        falconn.query2Layers_1D(MATRIX_Q, qParam.n_neighbors, qParam.verbose);
    }

    // 2D index
//    falconn.build2Layers(MATRIX_X);
    //    MATRIX_X.resize(0, 0); // Note: we should resize MatrixX for saving memory
//    falconn.setQueryParam(qParam.n_queries, qParam.n_neighbors, qParam.qProbes, qParam.n_threads, qParam.verbose);
//
//    for (int i = 1; i <= 10; ++i)
//    {
//        falconn.set_qProbes(1000 * i);
//        falconn.query2Layers(MATRIX_Q, qParam.n_neighbors, qParam.verbose);
//    }

    return 0;

    // MATRIX_Q.resize(0, 0); // Note: we should not resize if testing multiple qProbes
}
