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
MatrixXi BF::mips_topK(const Ref<const MatrixXf> & matQ, int n_neighbors)
{
    auto start = chrono::high_resolution_clock::now();

    int n_queries = matQ.cols();
    MatrixXi matTopK = MatrixXi::Zero(n_neighbors, n_queries); // K x Q

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(BF::n_threads);
#pragma omp parallel for
    for (int q = 0; q < n_queries; ++q)
    {

//        if (omp_get_num_threads() != num_threads)
//            abort();

        // Reset
        priority_queue<IFPair, vector<IFPair>, greater<IFPair>> queTopK;

        // Get query
        VectorXf vecQuery = matQ.col(q); // D x 1

        // Let Eigen decide SSE or AVX support
        VectorXf vecRes = vecQuery.transpose() * BF::matrix_X; // (1 x D) * (D x N)

        // cout << vecRes.maxCoeff() << endl;

        // Insert into priority queue to get TopK
        for (int n = 0; n < BF::n_points; ++n)
        {
            float fValue = vecRes(n);

            //cout << fValue << endl;

            // If we do not have enough top K point, insert
            if (int(queTopK.size()) < n_neighbors)
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

        IVector vecTopK(n_neighbors, -1); //init by -1
        for (int k = n_neighbors - 1; k >= 0; --k)
        {
            vecTopK[k] = queTopK.top().m_iIndex;
            queTopK.pop();
        }

        matTopK.col(q) = Map<VectorXi>(vecTopK.data(), n_neighbors);

    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "BF querying Time: " << (float)duration.count() << " ms" << endl;


    string sFileName = "BF_TopK_" + int2str(n_neighbors) + ".txt";
    outputFile(matTopK, sFileName);

    return matTopK.transpose();
}