#include "Utilities.h"
#include "Header.h"

#include <random>
#include <fstream> // fscanf, fopen, ofstream
#include <sstream>

/**
Generate dynamic bitset for HD3HD2HD1
We need to generate two bitsets for 2 LSH functions
**/
void bitHD3Generator2(int p_iNumBit)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_int_distribution<uint32_t> unifDist(0, 1);

    bitHD1 = boost::dynamic_bitset<> (p_iNumBit);
    bitHD2 = boost::dynamic_bitset<> (p_iNumBit);

    // Loop col first since we use col-wise
    for (int d = 0; d < p_iNumBit; ++d)
    {
        bitHD1[d] = unifDist(generator) & 1;
        bitHD2[d] = unifDist(generator) & 1;
    }
}

/**
Input:
(col-wise) matrix p_matKNN of size K x Q

Output: Q x K
- Each row is for each query
**/
void outputFile(const Ref<const MatrixXi> & p_matKNN, string p_sOutputFile)
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

void clearFalconnIndex()
{
    // 1D data structure
    VECTOR_PAIR_FALCONN_BUCKET_POS.clear();
    VECTOR_FALCONN_TABLES.clear();

    bitHD1.clear();
    bitHD2.clear();
}
