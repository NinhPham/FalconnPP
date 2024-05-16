//
// Created by npha145 on 17/03/24.
//

#include <omp.h>
#include "BF.h"
#include "Utilities.h"

/**
Retrieve n_neighbors MIPS entries using brute force computation

Input:
- MatrixXd: MATRIX_X (col-major) of size D X N
- MatrixXd: MATRIX_Q (col-major) of size D x Q

**/
MatrixXi BF::mips_topK()
{
    auto start = chrono::high_resolution_clock::now();

    MatrixXi matTopK = MatrixXi::Zero(BF::topK, BF::numQueries); // K x Q

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(BF::qThreads);
#pragma omp parallel for
    for (int q = 0; q < BF::numQueries; ++q)
    {

//        if (omp_get_num_threads() != num_threads)
//            abort();

        // Reset
        priority_queue<IFPair, vector<IFPair>, greater<IFPair>> queTopK;

        // Get query
        VectorXf vecQuery = BF::matrix_Q.col(q); // D x 1

        // Let Eigen decide SSE or AVX support
        VectorXf vecRes = vecQuery.transpose() * BF::matrix_X; // (1 x D) * (D x N)

        // cout << vecRes.maxCoeff() << endl;

        // Insert into priority queue to get TopK
        for (int n = 0; n < BF::numPoints; ++n)
        {
            float fValue = vecRes(n);

            //cout << fValue << endl;

            // If we do not have enough top K point, insert
            if (int(queTopK.size()) < BF::topK)
                queTopK.emplace(n, fValue);
            else // we have enough,
            {
                if (fValue > queTopK.top().m_fValue)
                {
                    queTopK.pop();
                    queTopK.emplace(n, fValue);
                }
            }
        }

        // Save result into mat_topK
        // Note that priorityQueue pop smallest element first

        IVector vecTopK(BF::topK, -1); //init by -1
        for (int k = BF::topK - 1; k >= 0; --k)
        {
            vecTopK[k] = queTopK.top().m_iIndex;
            queTopK.pop();
        }

        matTopK.col(q) = Map<VectorXi>(vecTopK.data(), BF::topK);

    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "BF querying Time: " << (float)duration.count() << " ms" << endl;


    string sFileName = "BF_TopK_" + int2str(BF::topK) + ".txt";
    outputFile(matTopK, sFileName);

    return matTopK.transpose();
}