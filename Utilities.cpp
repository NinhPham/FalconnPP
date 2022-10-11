#include "Utilities.h"
#include "Header.h"

#include <vector>
#include <queue>
#include <random>
#include <fstream> // fscanf, fopen, ofstream
#include <sstream>
#include <algorithm> // set_intersect(), lower_bound()
#include <unordered_map>
#include <unordered_set>

// #include "math.hpp" // do not use it any more

/**
    Fast in-place Walsh-Hadamard Transform (http://www.musicdsp.org/showone.php?id=18)
    also see (http://stackoverflow.com/questions/22733444/fast-sequency-ordered-walsh-hadamard-transform/22752430#22752430)
    - Note that the running time is exactly NlogN
**/
//void FWHT (Ref<VectorXf> data, const Ref<VectorXi> p_vecHD)
//{
//    //printVector(data);
//
//    int n = (int)data.size();
//    int nlog2 = log2(n);
//
//    int l, m;
//    for (int i = 0; i < nlog2; ++i)
//    {
//        l = 1 << (i + 1);
//        for (int j = 0; j < n; j += l)
//        {
//            m = 1 << i;
//            for (int k = 0; k < m; ++k)
//            {
//                //cout << data (j + k) << endl;
//                data (j + k) = data (j + k) * p_vecHD[j + k]; // There is a bug here
//                //cout << data (j + k) << endl;
//
//                //cout << data (j + k + m) << endl;
//                data (j + k + m) = data (j + k + m) * p_vecHD[j + k + m]; // There is a bug here
//                //cout << data (j + k + m) << endl;
//
//                wht_bfly (data (j + k), data (j + k + m));
//                //cout << data (j + k) << endl;
//                //cout << data (j + k + m) << endl;
//
//            }
//
//        }
//    }
//
//    //printVector(data);
//}


/* Generate Hadamard matrix
*/
//void generateHadamard(int p_iSize)
//{
//    MATRIX_HADAMARD = MatrixXi::Zero(PARAM_LSH_NUM_PROJECTION, PARAM_LSH_NUM_PROJECTION);
//    MATRIX_HADAMARD(0, 0) = 1;
//    int x, i, j;
//    for (x = 1; x < PARAM_LSH_NUM_PROJECTION; x += x)
//    {
//        for (i = 0; i < x; i++)
//        {
//            for (j = 0; j < x; j++)
//            {
//                MATRIX_HADAMARD(i + x, j) = MATRIX_HADAMARD(i, j);
//                /*cout << hadmard(i, j + x)<<endl;
//                cout << hadmard(i, j) << endl;*/
//                MATRIX_HADAMARD(i, j + x) = MATRIX_HADAMARD(i, j);
//                MATRIX_HADAMARD(i + x, j + x) = -MATRIX_HADAMARD(i, j);
//            }
//        }
//    }
//
//    /*
//    MatrixXi Hadamard3 = Hadamard * Hadamard * Hadamard;
//    VectorXi D4 = HD1.cwiseProduct(HD2.cwiseProduct(HD3));
//    for (int c = 0; c < PARAM_DATA_D_UP; c++)
//    	{
//        Hadamard3.col(c) = Hadamard3.col(c) * D4[c];
//    	}
//    return Hadamard3;
//    */
//}

void HD3Generator(int p_iRow, int p_iCol)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_int_distribution<uint32_t> unifDist(0, 1);

    MATRIX_HD1 = MatrixXf::Zero(p_iRow, p_iCol);

    // Loop col first since we use col-wise
    for (int l = 0; l < p_iCol; ++l)
    {
        for (int d = 0; d < p_iRow; ++d)
        {
            MATRIX_HD1(d, l) = 2.0 * unifDist(generator) - 1;
            // MATRIX_HD2(d, l) = 2 * unifDist(generator) - 1;
            // MATRIX_HD3(d, l) = 2 * unifDist(generator) - 1;
        }
    }
}

void HD3Generator2(int p_iRow, int p_iCol)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_int_distribution<uint32_t> unifDist(0, 1);

    MATRIX_HD1 = MatrixXf::Zero(p_iRow, p_iCol);
    MATRIX_HD2 = MatrixXf::Zero(p_iRow, p_iCol);

    // Loop col first since we use col-wise
    for (int l = 0; l < p_iCol; ++l)
    {
        for (int d = 0; d < p_iRow; ++d)
        {
            MATRIX_HD1(d, l) = 2.0 * unifDist(generator) - 1;
            MATRIX_HD2(d, l) = 2.0 * unifDist(generator) - 1;
        }
    }
}

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
Generate {+1, -1} vector
**/
VectorXi listHD3Generator(int p_iSize)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_int_distribution<uint32_t> unifDist(0, 1);

    VectorXi vecRandom = VectorXi::Zero(p_iSize);
    for (int d = 0; d < p_iSize; ++d)
    {
        vecRandom(d) = 2 * unifDist(generator) - 1;
    }

    return vecRandom;
}

void outputFile(const vector<vector<int>> &result, string p_sOutputFile)
{
//	cout << "Outputing File" << endl;
	ofstream myfile(p_sOutputFile);

	for (vector<int> KNN : result)
	{
		for (size_t k = 0; k < KNN.size(); ++k)
		{
			myfile << KNN[k] << ' ';
		}
		myfile << '\n';
	}

	myfile.close();
//	cout << "Done" << endl;
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


//VectorXi computeSimilarity(const set<int> &p_Candidates, int q)
//{
//
//	priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;
//
//	VectorXf vecQuery = MATRIX_Q.col(q);
//	for (int pointIdx : p_Candidates)
//	{
//		float fInnerProduct = vecQuery.dot(MATRIX_X.col(pointIdx));
//
//		if (int(minQueTopK.size()) < PARAM_MIPS_TOP_K)
//		{
//			minQueTopK.push(IFPair(pointIdx, fInnerProduct));
//		}
//		else
//		{
//			if (fInnerProduct > minQueTopK.top().m_fValue)
//            {
//                minQueTopK.pop();
//				minQueTopK.push(IFPair(pointIdx, fInnerProduct));
//            }
//		}
//	}
//
//    // saveQueue(minQueTopK, "Falconn2_" + int2str(q) + ".txt");
//
//    IVector vecTopK(PARAM_MIPS_TOP_K, -1); // assign -1 in case cause error
//    for (int k = PARAM_MIPS_TOP_K - 1; k >= 0; --k)
//    {
//        vecTopK[k] = minQueTopK.top().m_iIndex;
//        minQueTopK.pop();
//    }
//
//    printVector(vecTopK);
//
//	return Map<VectorXi>(vecTopK.data(), PARAM_MIPS_TOP_K);
//}

void getTopK(const boost::dynamic_bitset<> &setHist, const Ref<VectorXf>& p_vecQuery, Ref<VectorXi> p_vecTopK)
{
	vector<IFPair> vecDist(setHist.count());

	size_t n = 0;
    size_t index = setHist.find_first();
    while (index != boost::dynamic_bitset<>::npos)
    {
        /* do something */
        //float fInnerProduct = p_vecQuery.dot(MATRIX_X.col(index));
        vecDist[n] = IFPair(index, p_vecQuery.dot(MATRIX_X.col(index)));
        n++;
        index = setHist.find_next(index);
    }

    assert(n == setHist.count());

    // assert((int)vecDist.size() >= PARAM_MIPS_TOP_K);

    if ( (int)vecDist.size() > PARAM_MIPS_TOP_K )
    {
        // Get top k
        std::nth_element(vecDist.begin(), vecDist.begin() + PARAM_MIPS_TOP_K, vecDist.end(), greater<IFPair>());

        vector<IFPair> vecTopK;
        vecTopK.assign(vecDist.begin(), vecDist.begin() + PARAM_MIPS_TOP_K);
        sort(vecTopK.begin(), vecTopK.end(), greater<IFPair>());

        for (int k = 0; k < PARAM_MIPS_TOP_K; ++k)
        {
            p_vecTopK(k) = vecTopK[k].m_iIndex;
        }
    }
    else
    {
        sort(vecDist.begin(), vecDist.end(), greater<IFPair>());
        for (int k = 0; k < (int)vecDist.size(); ++k)
        {
            p_vecTopK(k) = vecDist[k].m_iIndex;
        }
    }


    /*
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;
	for (int n = 0; n < PARAM_DATA_N; ++n)
	{
        if (setHist[n])
        {
            float fInnerProduct = p_vecQuery.dot(MATRIX_X.col(n));
//            if (__AVX__)
//                fInnerProduct = InnerProductSIMD4ExtAVX(p_vecQuery.data(), MATRIX_X.col(n).data(), PARAM_DATA_D);
//            else
//                fInnerProduct = p_vecQuery.dot(MATRIX_X.col(n));

            if (int(minQueTopK.size()) < PARAM_MIPS_TOP_K)
                minQueTopK.push(IFPair(n, fInnerProduct));
            else
            {
                if (fInnerProduct > minQueTopK.top().m_fValue)
                {
                    minQueTopK.pop();
                    minQueTopK.push(IFPair(n, fInnerProduct));
                }
            }
        }

	}
    //	assert((int)minQueTopK.size() == PARAM_MIPS_TOP_K);

    for (int k = PARAM_MIPS_TOP_K - 1; k >= 0; --k)
    {
        p_vecTopK(k) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
    */
}

void getTopK(const Ref<VectorXf>& p_vecHist, const Ref<VectorXf>& p_vecQuery, Ref<VectorXi> p_vecTopK)
{
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopB;
	for (int n = 0; n < PARAM_DATA_N; ++n)
	{
        if (p_vecHist[n] > 0.0)
        {
            //float fInnerProduct = p_vecQuery.dot(MATRIX_X.col(n));
//            if (__AVX__)
//                fInnerProduct = InnerProductSIMD4ExtAVX(p_vecQuery.data(), MATRIX_X.col(n).data(), PARAM_DATA_D);
//            else
//                fInnerProduct = p_vecQuery.dot(MATRIX_X.col(n));

            if (int(minQueTopB.size()) < PARAM_MIPS_CANDIDATE_SIZE)
                minQueTopB.push(IFPair(n, p_vecHist[n]));
            else
            {
                if (p_vecHist[n] > minQueTopB.top().m_fValue)
                {
                    minQueTopB.pop();
                    minQueTopB.push(IFPair(n, p_vecHist[n]));
                }
            }
        }

	}

	priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;
	while (!minQueTopB.empty())
    {
        IFPair ifPair = minQueTopB.top();
        minQueTopB.pop();

        int n = ifPair.m_iIndex;
        float fInnerProduct = p_vecQuery.dot(MATRIX_X.col(n));
//            if (__AVX__)
//                fInnerProduct = InnerProductSIMD4ExtAVX(p_vecQuery.data(), MATRIX_X.col(n).data(), PARAM_DATA_D);
//            else
//                fInnerProduct = p_vecQuery.dot(MATRIX_X.col(n));

        if (int(minQueTopK.size()) < PARAM_MIPS_TOP_K)
            minQueTopK.push(IFPair(n, fInnerProduct));
        else
        {
            if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(n, fInnerProduct));
            }
        }

    }

//    assert((int)minQueTopK.size() == PARAM_MIPS_TOP_K);

    for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
    {
        p_vecTopK(k) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

}

/** \brief Return top K from counting histogram
 *
 * \param
 *
 - vecCounter: counting histogram of 1 x N
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - vector<int> contains top K point indexes with largest values
 - We need to sort based on the value of the histogram.
 *
 */
void getTopK(const unordered_map<int, float> &mapCounter, const Ref<VectorXf>& p_vecQuery, Ref<VectorXi> p_vecTopK)
{

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopB;

    // iterate hashMap
    for (const auto& kv : mapCounter)
    {
        if (int(minQueTopB.size()) < PARAM_MIPS_CANDIDATE_SIZE)
            minQueTopB.push(IFPair(kv.first, kv.second));

        else if (kv.second > minQueTopB.top().m_fValue)
        {
            minQueTopB.pop();
            minQueTopB.push(IFPair(kv.first, kv.second));
        }
    }

//    assert((int)minQueTopB.size() == PARAM_MIPS_CANDIDATE_SIZE);

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;
	while (!minQueTopB.empty())
    {
        IFPair ifPair = minQueTopB.top();
        minQueTopB.pop();

        int n = ifPair.m_iIndex;
        float fInnerProduct = p_vecQuery.dot(MATRIX_X.col(n));
//            if (__AVX__)
//                fInnerProduct = InnerProductSIMD4ExtAVX(p_vecQuery.data(), MATRIX_X.col(n).data(), PARAM_DATA_D);
//            else
//                fInnerProduct = p_vecQuery.dot(MATRIX_X.col(n));

        if (int(minQueTopK.size()) < PARAM_MIPS_TOP_K)
            minQueTopK.push(IFPair(n, fInnerProduct));
        else
        {
            if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(n, fInnerProduct));
            }
        }

    }

//    assert((int)minQueTopK.size() == PARAM_MIPS_TOP_K);

    for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
    {
        p_vecTopK(k) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
 }

 /** \brief Compute topB and then B inner product to return topK
 *
 * \param
 *
 - mapCounter: key = pointIdx, value = partialEst of Inner Product
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - VectorXi::p_vecTopK contains top K point indexes with largest values
 *
 */
void extract_TopB_TopK_Histogram(const unordered_map<int, float> &mapCounter, const Ref<VectorXf> &p_vecQuery,
                                 int p_iTopB, int p_iTopK, Ref<VectorXi> p_vecTopK)
{
    // Find topB
    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    // cout << "Number of unique candidates: " << mapCounter.size() << endl;
//    assert((int)mapCounter.size() >= p_iTopK);
    p_iTopB = min((int)mapCounter.size(), p_iTopB);

    for (const auto& kv : mapCounter) // access via kv.first, kv.second
    {
        if ((int)minQueTopK.size() < p_iTopB)
            minQueTopK.push(IFPair(kv.first, kv.second));
        else if (kv.second > minQueTopK.top().m_fValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IFPair(kv.first, kv.second));
        }
    }



    // The largest value should come first.
    IVector vecTopB(p_iTopB);
    for (int n = p_iTopB - 1; n >= 0; --n)
    {
        // Get point index
        vecTopB[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    // Might need to sort it so that it would be cache-efficient when computing dot product
    // If B is small then might not have any effect
    sort(vecTopB.begin(), vecTopB.end()); // matter if B is large

    // Find top-K
    //for (int n = 0; n < (int)vecTopB.size(); ++n)
    for (const auto& iPointIdx: vecTopB)
    {
        // Get point Idx
        //int iPointIdx = vecTopB[n];
        float fValue = p_vecQuery.dot(MATRIX_X.col(iPointIdx));

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(iPointIdx, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(iPointIdx, fValue));
            }
        }
    }



    for (int n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        p_vecTopK(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
}

/** \brief Compute topB and then B inner product to return topK
 *
 * \param
 *
 - vecCounter: Inner product estimation histogram of N x 1
 - p_iTopK: top K largest values
 *
 *
 * \return
 *
 - VectorXi::p_vecTopK contains top K point indexes with largest values
 *
 */
void extract_TopB_TopK_Histogram(const Ref<VectorXf> & p_vecCounter, const Ref<VectorXf> &p_vecQuery,
                                 int p_iTopB, int p_iTopK, Ref<VectorXi> p_vecTopK)
{
    // Find topB
//    assert((int)p_vecCounter.size() >= p_iTopK);

    priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

    for (int n = 0; n < (int)p_vecCounter.size(); ++n)
    {
        float fTemp = p_vecCounter(n);

        if ((int)minQueTopK.size() < p_iTopB)
            minQueTopK.push(IFPair(n, fTemp));
        else if (fTemp > minQueTopK.top().m_fValue)
        {
            minQueTopK.pop();
            minQueTopK.push(IFPair(n, fTemp));
        }
    }

    // The largest value should come first.
    IVector vecTopB(p_iTopB);
    for (int n = p_iTopB - 1; n >= 0; --n)
    {
        // Get point index
        vecTopB[n] = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }

    // Might need to sort it so that it would be cache-efficient when computing dot product
    // If B is small then might not have any effect
    sort(vecTopB.begin(), vecTopB.end());

    // Find top-K
    //for (int n = 0; n < (int)vecTopB.size(); ++n)
    for (const auto & iPointIdx: vecTopB)
    {
        // Get point Idx
        // int iPointIdx = vecTopB[n];
        float fValue = p_vecQuery.dot(MATRIX_X.col(iPointIdx));

        // Insert into minQueue
        if ((int)minQueTopK.size() < p_iTopK)
            minQueTopK.push(IFPair(iPointIdx, fValue));
        else
        {
            // Insert into minQueue
            if (fValue > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(iPointIdx, fValue));
            }
        }
    }


    for (int n = p_iTopK - 1; n >= 0; --n)
    {
        // Get point index
        p_vecTopK(n) = minQueTopK.top().m_iIndex;
        minQueTopK.pop();
    }
}

void clearFalconnIndex()
{
    VECTOR2D_FALCONN_TABLES.clear();
    VECTOR2D_PAIR_FALCONN_TABLES.clear();
    VECTOR3D_FALCONN_TABLES.clear();

    // 1D data structure
    VECTOR_PAIR_FALCONN_BUCKET_POS.clear();
    VECTOR_FALCONN_TABLES.clear();
    VECTOR_PAIR_FALCONN_TABLES.clear();

    MATRIX_HD1.resize(0, 0);
    MATRIX_HD2.resize(0, 0);
}
