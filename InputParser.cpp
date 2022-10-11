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
        iType = 11;
        cout << "Bruteforce topK... " << endl;
        cout << endl;
    }

    // Falconn++
    else if (strcmp(args[7], "FalconnCEOs2") == 0)
    {
        iType = 21;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Index probes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[11]);
        cout << "Query probes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        PARAM_LSH_BUCKET_SIZE_SCALE = atof(args[12]);
        cout << "Bucket size scale: " << PARAM_LSH_BUCKET_SIZE_SCALE << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[13]);
        cout << "Candidate size: " << PARAM_MIPS_CANDIDATE_SIZE << endl;
    }

    // Falconn
    else if (strcmp(args[7], "Falconn") == 0)
    {
        iType = 31;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[10]);
        cout << "Query probes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[11]);
        cout << "Candidate size: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = atoi(args[12]);
//        if (PARAM_LSH_PROBING_HEURISTIC == 1)
//            cout << "Using CEOs heuristic: probing sequence based on the abs projection value." << endl;
//        else
//            cout << "Using Falconn heuristic: probing sequence based on the projection distance." << endl;

//        cout << endl;
    }

    // Falconn++
    else if (strcmp(args[7], "FalconnCEOs") == 0)
    {
        iType = 32;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Index probes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[11]);
        cout << "Query probes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        float fScale = atof(args[12]);
        if (fScale > 1.0)
        {
            PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = true;
            PARAM_LSH_BUCKET_SIZE_LIMIT = (int)fScale;
            cout << "Bucket size limit: " << PARAM_LSH_BUCKET_SIZE_LIMIT << endl;
        }
        else if (fScale > 0.0)
        {
            PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = false;
            PARAM_LSH_BUCKET_SIZE_SCALE = fScale;
            cout << "Bucket size scale: " << PARAM_LSH_BUCKET_SIZE_SCALE << endl;
        }

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[13]);
        cout << "Candidate size: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = atoi(args[14]);
//        if (PARAM_LSH_PROBING_HEURISTIC == 1)
//            cout << "Using CEOs heuristic: probing sequence based on the abs projection value." << endl;
//        else
//            cout << "Using Falconn heuristic: probing sequence based on the projection distance." << endl;

//        cout << endl;
    }
    else if (strcmp(args[7], "cutFalconnCEOs") == 0)
    {
        iType = 33;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Index probes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[11]);
        cout << "Query probes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        PARAM_LSH_DISCARD_T = atof(args[12]);
        cout << "Discarding threshold: " << PARAM_LSH_DISCARD_T << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[13]);
        cout << "Candidate size: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = atoi(args[14]);

    }
    else if (strcmp(args[7], "FalconnCEOs_Est") == 0)
    {
        iType = 34;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Index probes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[11]);
        cout << "Query probes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        PARAM_LSH_BUCKET_SIZE_SCALE = atof(args[12]);

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[13]);
        cout << "Candidate size: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = atoi(args[14]);
//        if (PARAM_LSH_PROBING_HEURISTIC == 1)
//            cout << "Using CEOs heuristic: probing sequence based on the abs projection value." << endl;
//        else
//            cout << "Using Falconn heuristic: probing sequence based on the projection distance." << endl;

//        cout << endl;
    }
    else if (strcmp(args[7], "FalconnCEOsStream") == 0)
    {
        iType = 35;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Index probes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[11]);
        cout << "Query probes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        PARAM_LSH_BUCKET_SIZE_LIMIT = atoi(args[12]);
        cout << "Bucket size limit: " << PARAM_LSH_BUCKET_SIZE_LIMIT << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[13]);
        cout << "Candidate size: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = atoi(args[14]); // only 1 or 0
//        if (PARAM_LSH_PROBING_HEURISTIC == 1)
//            cout << "Using CEOs heuristic: probing sequence based on the abs projection value." << endl;
//        else
//            cout << "Using Falconn heuristic: probing sequence based on the projection distance." << endl;

        cout << endl;
    }
    else if (strcmp(args[7], "FalconnCEOsEst") == 0)
    {
        iType = 36;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of hash functions: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Index probes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[11]);
        cout << "Query probes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        PARAM_LSH_BUCKET_SIZE_LIMIT = atoi(args[12]);
        cout << "Bucket size limit: " << PARAM_LSH_BUCKET_SIZE_LIMIT << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[13]);
        cout << "Candidate size: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        cout << endl;
    }

    else if (strcmp(args[7], "test_Scale") == 0)
    {
        cout << "Fix upD, qProbes, L, iProbes, varying scale..." << endl;
        iType = 999;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Number of iProbes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[11]);
        cout << "Number of qProbes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[12]);

        if (PARAM_MIPS_CANDIDATE_SIZE == 0)
            PARAM_MIPS_CANDIDATE_SIZE = PARAM_DATA_N;

        cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = 1;
        PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = false;

    }
    else if (strcmp(args[7], "test1D_Scale") == 0)
    {
        cout << "Fix upD, qProbes, L, iProbes, varying scale..." << endl;
        iType = 998;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Number of iProbes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[11]);
        cout << "Number of qProbes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[12]);

        if (PARAM_MIPS_CANDIDATE_SIZE == 0)
            PARAM_MIPS_CANDIDATE_SIZE = PARAM_DATA_N;

        cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = 1;
        PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = false;
    }
    else if (strcmp(args[7], "test1D_Thres") == 0)
    {
        cout << "Fix upD, qProbes, L, iProbes, varying scale..." << endl;
        iType = 997;

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

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[12]);

        if (PARAM_MIPS_CANDIDATE_SIZE == 0)
            PARAM_MIPS_CANDIDATE_SIZE = PARAM_DATA_N;

        cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = 1;
        PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = false;
    }

    else if (strcmp(args[7], "test_Scale_qProbe") == 0)
    {
        cout << "Fix upD, L, iProbes, varying scale and qProbes..." << endl;
        iType = 996;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Number of iProbes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_TEST_LSH_qPROBE_BASE = atoi(args[11]);
        cout << "Number of base qProbes: " << PARAM_TEST_LSH_qPROBE_BASE << endl;

        PARAM_TEST_LSH_qPROBE_RANGE = atoi(args[12]);
        cout << "Number of range qProbes: " << PARAM_TEST_LSH_qPROBE_RANGE << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[13]);

        if (PARAM_MIPS_CANDIDATE_SIZE == 0)
            PARAM_MIPS_CANDIDATE_SIZE = PARAM_DATA_N;

        cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;


        PARAM_LSH_PROBING_HEURISTIC = 1;
        PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = false;
    }
    else if (strcmp(args[7], "test_Est_Scale") == 0)
    {
        cout << "Fix upD, qProbes, L, iProbes, varying scale and qProbes..." << endl;
        iType = 995;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Number of iProbes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_LSH_NUM_QUERY_PROBES = atoi(args[11]);
        cout << "Number of qProbes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[12]);
        cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = 1;
        PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = false;

    }

    else if (strcmp(args[7], "test_Est_Scale_qProbe") == 0)
    {
        cout << "Fix upD, qProbes, L, iProbes, varying scale and qProbes..." << endl;
        iType = 994;

        PARAM_LSH_NUM_PROJECTION = atoi(args[8]);
        cout << "Number of projections: " << PARAM_LSH_NUM_PROJECTION << endl;

        PARAM_LSH_NUM_TABLE = atoi(args[9]);
        cout << "Number of hash tables: " << PARAM_LSH_NUM_TABLE << endl;

        PARAM_LSH_NUM_INDEX_PROBES = atoi(args[10]);
        cout << "Number of iProbes: " << PARAM_LSH_NUM_INDEX_PROBES << endl;

        PARAM_TEST_LSH_qPROBE_BASE = atoi(args[11]);
        cout << "Number of base qProbes: " << PARAM_TEST_LSH_qPROBE_BASE << endl;

        PARAM_TEST_LSH_qPROBE_RANGE = atoi(args[12]);
        cout << "Number of range qProbes: " << PARAM_TEST_LSH_qPROBE_RANGE << endl;

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[13]);
        cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = 1;
        PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = false;

    }

    else if (strcmp(args[7], "test_scaledIndex_1D") == 0)
    {
        cout << "Fix upD, L, iProbes, scaled, varying qProbes..." << endl;
        iType = 993;

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

        PARAM_MIPS_CANDIDATE_SIZE = atoi(args[14]);

        if (PARAM_MIPS_CANDIDATE_SIZE == 0)
            PARAM_MIPS_CANDIDATE_SIZE = PARAM_DATA_N;

        cout << "Number of inner product computations: " << PARAM_MIPS_CANDIDATE_SIZE << endl;

        PARAM_LSH_PROBING_HEURISTIC = 1;
        PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE = false;

    }

    // Setting internal parameter
    PARAM_INTERNAL_LSH_NUM_BUCKET = PARAM_LSH_NUM_PROJECTION * 2;
    PARAM_INTERNAL_LOG2_NUM_PROJECTION = log2(PARAM_LSH_NUM_PROJECTION);
    PARAM_INTERNAL_LOG2_FWHT_PROJECTION = log2(PARAM_INTERNAL_FWHT_PROJECTION);

    return iType;
}

