#include "InputParser.h"
#include "Header.h"

#include <stdlib.h>     /* atoi */
#include <iostream> // cin, cout
#include <fstream> // fscanf, fopen, ofstream

#include <vector>
#include <string.h> // strcmp

int loadInput(int nargs, char** args)
{
    if (nargs < 6)
        exit(1);

    // Parse arguments: Note that don't know what args[0] represents for !!!
    PARAM_DATA_N = atoi(args[1]);
//    cout << "Number of rows of X: " << PARAM_DATA_N << endl;

    PARAM_QUERY_Q = atoi(args[2]);
//    cout << "Number of rows of Q: " << PARAM_QUERY_Q << endl;

    PARAM_DATA_D = atoi(args[3]);
//    cout << "Number of dimensions: " << PARAM_DATA_D << endl;

    PARAM_MIPS_TOP_K = atoi(args[4]);
//    cout << "Top K: " << PARAM_MIPS_TOP_K << endl;
//    cout << endl;

    /** NOTE: It seems that reading IO is not safe for OpenMP **/

    // Read the row-wise matrix X, and convert to col-major Eigen matrix
//    cout << "Read row-wise X, it will be converted to col-major Eigen matrix of size D x N..." << endl;
    if (args[5])
    {
        FILE *f = fopen(args[5], "r");
        if (!f)
        {
            printf("Data file does not exist");
            exit(1);
        }

        FVector vecTempX(PARAM_DATA_D * PARAM_DATA_N, 0.0);

        // Each line is a vector of D dimensions
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            for (int d = 0; d < PARAM_DATA_D; ++d)
            {
                fscanf(f, "%f", &vecTempX[n * PARAM_DATA_D + d]);
                // cout << vecTempX[n + d * PARAM_DATA_N] << " ";
            }
            // cout << endl;
        }

        // Matrix_X is col-major
        MATRIX_X = Map<MatrixXf>(vecTempX.data(), PARAM_DATA_D, PARAM_DATA_N);
//        MATRIX_X.transpose();
//        cout << "X has " << MATRIX_X.rows() << " rows and " << MATRIX_X.cols() << " cols " << endl;

        /**
        Print the first col (1 x N)
        Print some of the first elements of the MATRIX_X to see that these elements are on consecutive memory cell.
        **/
//        cout << MATRIX_X.col(0) << endl << endl;
//        cout << "In memory (col-major):" << endl;
//        for (n = 0; n < 10; n++)
//            cout << *(MATRIX_X.data() + n) << "  ";
//        cout << endl << endl;

    }

    /**
    We center MATRIX_X
    **/
    VectorXf vecCenter = MATRIX_X.rowwise().sum(); // sum of all columns
//    cout << vecCenter.rows() << endl;
    vecCenter = vecCenter / PARAM_DATA_N;
    MATRIX_X = MATRIX_X.colwise() - vecCenter; // subtract each column by a vector

    // Read the row-wise matrix X, and convert to col-major Eigen matrix
//    cout << "Read row-wise Q, it will be converted to col-major Eigen matrix of size D x Q..." << endl;
    if (args[6])
    {
        FILE *f = fopen(args[6], "r+");
        if (!f)
        {
            printf("Data file does not exist");
            exit(1);
        }

        FVector vecTempQ(PARAM_DATA_D * PARAM_QUERY_Q, 0.0);

        for (int q = 0; q < PARAM_QUERY_Q; ++q)
        {
            for (int d = 0; d < PARAM_DATA_D; ++d)
            {
                fscanf(f, "%f", &vecTempQ[q * PARAM_DATA_D + d]); //%f = float, %lf: double
                //cout << vecTempQ[q * D + d] << " ";
            }
            //cout << endl;
        }

        MATRIX_Q = Map<MatrixXf>(vecTempQ.data(), PARAM_DATA_D, PARAM_QUERY_Q);
//        cout << "Q has " << MATRIX_Q.rows() << " rows and " << MATRIX_Q.cols() << " cols." << endl;

        /**
        Print the first row (1 x Q)
        Print some of the first elements of the MATRIX_Q to see that these elements are on consecutive memory cell.
        **/
//
//        cout << MATRIX_Q.col(0) << endl << endl;
//        cout << "In memory (col-major):" << endl;
//        for (n = 0; n < 10; n++)
//            cout << *(MATRIX_Q.data() + n) << "  ";
//        cout << endl << endl;

    }

//    cout << endl;


    // Algorithm
    int iType = 0;

    // Exact solution
    if (strcmp(args[7], "BF") == 0)
    {
        iType = 1;
        cout << "Bruteforce topK... " << endl;

        PARAM_OUTPUT_FILE = args[8];

        if (PARAM_OUTPUT_FILE.empty())
            PARAM_INTERNAL_SAVE_OUTPUT = false;
        else
            PARAM_INTERNAL_SAVE_OUTPUT = true;

        cout << endl;
    }

    // Falconn++
    else if (strcmp(args[7], "Falconn++") == 0)
    {
        iType = 2;

        cout << "Falconn++ topK... " << endl;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        if (PARAM_LSH_NUM_PROJECTION < PARAM_DATA_D)
            PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
        else
            PARAM_INTERNAL_FWHT_PROJECTION = PARAM_LSH_NUM_PROJECTION;

        cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        int iTemp = atoi(args[10]);
        PARAM_LSH_BUCKET_SIZE_SCALE = iTemp * 1.0 / 100;
        cout << "Bucket scale: " << PARAM_LSH_BUCKET_SIZE_SCALE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[11]);
        cout << "Index probes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        int iProbes = round(PARAM_MIPS_TOP_K * 4.0 * PARAM_LSH_NUM_PROJECTION * PARAM_LSH_NUM_PROJECTION / PARAM_DATA_N);
        iProbes = max(1, iProbes);
        cout << "Recommended index probes: " << iProbes << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[12]);
        cout << "Query probes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

//        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[13]);
//
//        if (PARAM_MIPS_CANDIDATE_SIZE == 0)
//            PARAM_MIPS_CANDIDATE_SIZE = PARAM_DATA_N;

//        cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        // None: No save file
        if (strcmp(args[13], "NoOutput") == 0)
            PARAM_INTERNAL_SAVE_OUTPUT = false;
        else
        {
            PARAM_INTERNAL_SAVE_OUTPUT = true;
            PARAM_OUTPUT_FILE = args[13];
        }

        // This parameter is important since it will limit the scaling of small buckets
        // Default is true
        // If false, then it use exactly \alpha * n (points) in each table
        // Setting false to compare with Falconn, and threshold LSF
        PARAM_INTERNAL_LIMIT_BUCKET = true;
    }

    else if (strcmp(args[7], "test_ScaledIndex_1D") == 0)
    {
        cout << "Fix upD, L, scale, iProbes, varying qProbes..." << endl;
        iType = 91;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        if (PARAM_LSH_NUM_PROJECTION < PARAM_DATA_D)
            PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
        else
            PARAM_INTERNAL_FWHT_PROJECTION = PARAM_LSH_NUM_PROJECTION;

        cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        int iTemp = atoi(args[10]);
        PARAM_LSH_BUCKET_SIZE_SCALE = iTemp * 1.0 / 100;
        cout << "Bucket scale: " << PARAM_LSH_BUCKET_SIZE_SCALE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[11]);
        cout << "Number of iProbes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_TEST_LSH_qPROBE_BASE = atoi(args[12]);
        cout << "Number of base qProbes: " << PARAM_TEST_LSH_qPROBE_BASE << endl;

        PARAM_TEST_LSH_qPROBE_RANGE = atoi(args[13]);
        cout << "Number of range qProbes: " << PARAM_TEST_LSH_qPROBE_RANGE << endl;

//        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[14]);
//
//        if (PARAM_MIPS_CANDIDATE_SIZE == 0)
//            PARAM_MIPS_CANDIDATE_SIZE = PARAM_DATA_N;

//        cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        if (strcmp(args[14], "NoOutput") == 0)
            PARAM_INTERNAL_SAVE_OUTPUT = false;
        else
        {
            PARAM_INTERNAL_SAVE_OUTPUT = true;
            PARAM_OUTPUT_FILE = args[14];
        }

        // This parameter is important since it will limit the scaling of small buckets
        // Default is true
        // If false, then it use exactly \alpha * n (points) in each table
        // Setting false to compare with Falconn, and threshold LSF
        PARAM_INTERNAL_LIMIT_BUCKET = true;
    }

    else if (strcmp(args[7], "test_ThresIndex_1D") == 0)
    {
        cout << "Fix upD, qProbes, L, iProbes, varying scale..." << endl;
        iType = 92;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        if (PARAM_LSH_NUM_PROJECTION < PARAM_DATA_D)
            PARAM_INTERNAL_FWHT_PROJECTION = 1 << int(ceil(log2(PARAM_DATA_D)));
        else
            PARAM_INTERNAL_FWHT_PROJECTION = PARAM_LSH_NUM_PROJECTION;

        cout << "Number of projections for FWHT: " << PARAM_INTERNAL_FWHT_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_TEST_LSH_qPROBE_BASE = atoi(args[10]);
        cout << "Number of base qProbes: " << PARAM_TEST_LSH_qPROBE_BASE << endl;

        PARAM_TEST_LSH_qPROBE_RANGE = atoi(args[11]);
        cout << "Number of range qProbes: " << PARAM_TEST_LSH_qPROBE_RANGE << endl;

        if (strcmp(args[12], "NoOutput")  == 0)
            PARAM_INTERNAL_SAVE_OUTPUT = false;
        else
        {
            PARAM_INTERNAL_SAVE_OUTPUT = true;
            PARAM_OUTPUT_FILE = args[12];
        }

    }

    // Setting internal parameter
    PARAM_INTERNAL_LSH_NUM_BUCKET = 2 * PARAM_LSH_NUM_PROJECTION;
    PARAM_INTERNAL_LOG2_NUM_PROJECTION = log2(PARAM_LSH_NUM_PROJECTION);
    PARAM_INTERNAL_LOG2_FWHT_PROJECTION = log2(PARAM_INTERNAL_FWHT_PROJECTION);

    return iType;
}

