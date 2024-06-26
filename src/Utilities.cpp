#include "Utilities.h"
#include "Header.h"

#include <fstream> // fscanf, fopen, ofstream

/**
Input:
(col-wise) matrix p_matKNN of size K x Q

Output: Q x K
- Each row is for each query
**/
void outputFile(const Ref<const MatrixXi> & p_matKNN, const string& p_sOutputFile)
{
//	cout << "Outputing File..." << endl;
	ofstream myfile(p_sOutputFile);

	//cout << p_matKNN << endl;

	for (int j = 0; j < p_matKNN.cols(); ++j)
	{
        //cout << "Print col: " << i << endl;
		for (int i = 0; i < p_matKNN.rows(); ++i)
		{
            myfile << p_matKNN(i, j) << ' ';

		}
		myfile << '\n';
	}

	myfile.close();
//	cout << "Done" << endl;
}

/**
 *
 * @param dataset
 * @param numPoints
 * @param numDim
 * @param MATRIX_X
 */
void loadtxtData(const string & dataset, int numPoints, int numDim, MatrixXf & MATRIX_X)
{
    FILE *f = fopen(dataset.c_str(), "r");
    if (!f) {
        cerr << "Error: Data file does not exist !" << endl;
        exit(1);
    }

    // Important: If use a temporary vector to store data, then it doubles the memory
    MATRIX_X = MatrixXf::Zero(numDim, numPoints);

    // Each line is a vector of D dimensions
    for (int n = 0; n < numPoints; ++n) {
        for (int d = 0; d < numDim; ++d) {
            fscanf(f, "%f", &MATRIX_X(d, n));
        }
    }

    cout << "Finish reading data" << endl;
}

/*
 * @param nargs:
 * @param args:
 * @return: Parsing parameter for FalconnPP++
 */
void readIndexParam(int nargs, char** args, IndexParam& iParam)
{
    if (nargs < 6)
        exit(1);

    // NumPoints n
    bool bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_points") == 0)
        {
            iParam.n_points = atoi(args[i + 1]);
            cout << "Number of rows/points of X: " << iParam.n_points << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Number of rows/points is missing !" << endl;
        exit(1);
    }

    // Dimension
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_features") == 0)
        {
            iParam.n_features = atoi(args[i + 1]);
            cout << "Number of columns/dimensions: " << iParam.n_features << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        cerr << "Error: Number of columns/dimensions is missing !" << endl;
        exit(1);
    }

    // Top-K: We need this param for indexing as we might NOT filter out points on the sparse buckets.
    // This param should be equal to top-K.
    // Otherwise, PARAM_BUCKET_TOP_K = 50 might suffice for top-K = {1, ..., 100})
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--bucket_minSize") == 0)
        {
            iParam.bucket_minSize = atoi(args[i + 1]);
            cout << "Minimum number of points in a bucket: " << iParam.bucket_minSize << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.bucket_minSize = 50;
        cout << "Default minimum number of points in a bucket: " << iParam.bucket_minSize << endl;
    }

    // n_tables
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_tables") == 0)
        {
            iParam.n_tables = atoi(args[i + 1]);
            cout << "Number of LSH tables: " << iParam.n_tables << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.n_tables = 10;
        cout << "Default number of LSH tables: " << iParam.n_tables << endl;
    }

    // numProjections
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_proj") == 0)
        {
            iParam.n_proj = atoi(args[i + 1]);
            cout << "Number of projections: " << iParam.n_proj << endl;
            bSuccess = true;
            break;

        }
    }
    if (!bSuccess)
    {
        int iTemp = ceil(log2(1.0 * iParam.n_features));
        iParam.n_proj = max(256, 1 << iTemp);
        cout << "Number of projections: " << iParam.n_proj << endl;
    }

    // index probing
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--iProbes") == 0)
        {
            iParam.iProbes = atoi(args[i + 1]);
            cout << "Number of index probes: " << iParam.iProbes << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.iProbes = 1;
        cout << "Default index probes: " << iParam.iProbes << endl;
    }

    // scaling bucket
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--bucket_scale") == 0)
        {
            iParam.bucket_scale = atof(args[i + 1]);
            cout << "Bucket size scale: " << iParam.bucket_scale << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.bucket_scale = 1.0;
        cout << "Default bucket scale: " << iParam.bucket_scale << endl;
    }

    // min size of bucket
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--bucket_minSize") == 0)
        {
            iParam.bucket_minSize = atoi(args[i + 1]);
            cout << "Bucket min size: " << iParam.bucket_minSize << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.bucket_minSize = 50;
        cout << "Default bucket min size: " << iParam.bucket_minSize << endl;
    }

    // n_threads
    iParam.n_threads = -1;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_threads") == 0)
        {
            iParam.n_threads = atoi(args[i + 1]);
            cout << "Number of threads: " << iParam.n_threads << endl;
            break;
        }
    }
    if (!bSuccess)
    {
        iParam.n_threads = -1;
        cout << "Use all threads: " << iParam.n_threads << endl;
    }

    // n_threads
    iParam.seed = -1;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--random_seed") == 0)
        {
            iParam.seed = atoi(args[i + 1]);
            cout << "Random seed: " << iParam.seed << endl;
            break;
        }
    }
    if (!bSuccess)
    {
        cout << "Use a random seed !" << endl;
    }

    // Top-s0 for for each layers
//    PARAM_LSH_SIMHASH_LENGTH = 0;
//    for (int i = 1; i < nargs; i++)
//    {
//        if (strcmp(args[i], "--numCodes") == 0)
//        {
//            PARAM_LSH_SIMHASH_LENGTH = atoi(args[i + 1]);
//            cout << "Number of 64-bit integers for SimHash : " << PARAM_LSH_SIMHASH_LENGTH << endl;
//            break;
//        }
//    }
}

/*
 * @param nargs:
 * @param args:
 * @return: Parsing parameter for FalconnPP++
 */
void readQueryParam(int nargs, char** args, QueryParam & qParam)
{
    if (nargs < 4)
        exit(1);

    // NumPoints n
    bool bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_queries") == 0)
        {
            qParam.n_queries = atoi(args[i + 1]);
            cout << "Number of rows/points of Q: " << qParam.n_queries << endl;
            bSuccess = true;
            break;
        }
    }

    if (!bSuccess)
    {
        cerr << "Error: Number of queries is missing !" << endl;
        exit(1);
    }

    // Qery dimension = Point dimensions

    // Top-K: We need this param for indexing as we might NOT filter out points on the sparse buckets.
    // This param should be equal to top-K.
    // Otherwise, PARAM_BUCKET_TOP_K = 50 might suffice for top-K = {1, ..., 100})
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--n_neighbors") == 0)
        {
            qParam.n_neighbors = atoi(args[i + 1]);
            cout << "Top-K query: " << qParam.n_neighbors << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        qParam.n_neighbors = 1;
        cout << "Default top-K: " << qParam.n_neighbors << endl;
    }



    // query probing
    bSuccess = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--qProbes") == 0)
        {
            qParam.qProbes = atoi(args[i + 1]);
            cout << "Number of query probes: " << qParam.qProbes << endl;
            bSuccess = true;
            break;
        }
    }
    if (!bSuccess)
    {
        qParam.qProbes = 10;
        cout << "Default query probes: " << qParam.qProbes << endl;
    }

    // Candidate size
//    bSuccess = false;
//    for (int i = 1; i < nargs; i++)
//    {
//        if (strcmp(args[i], "--cand") == 0)
//        {
//            PARAM_MIPS_CANDIDATE_SIZE = atoi(args[i + 1]);
//            cout << "Number of candidates: " << PARAM_MIPS_CANDIDATE_SIZE << endl;
//            bSuccess = true;
//            break;
//        }
//    }
//    if (!bSuccess)
//    {
//        PARAM_MIPS_CANDIDATE_SIZE = PARAM_DATA_N;
//        cout << "Default number of candidates: " << PARAM_MIPS_CANDIDATE_SIZE << endl;
//    }

    // verbose
    qParam.verbose = false;
    for (int i = 1; i < nargs; i++)
    {
        if (strcmp(args[i], "--verbose") == 0)
        {
            qParam.verbose = true;
            cout << "Verbose mode: " << qParam.verbose << endl;
            break;
        }
    }

    // Use CEOs to estimate inner product
//    PARAM_QUERY_DOT_ESTIMATE = false;
//    for (int i = 1; i < nargs; i++)
//    {
//        if (strcmp(args[i], "--useEst") == 0)
//        {
//            PARAM_QUERY_DOT_ESTIMATE = true;
//            cout << "Use CEOs to estimate inner product." << endl;
//            break;
//        }
//    }

}