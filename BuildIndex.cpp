#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <unordered_map>
#include <queue>

#include "Header.h"
#include "Utilities.h"
#include "BuildIndex.h"

/**
Falconn++: use CEOs to filter out points in the bucket
- We use 2 concatenating LSH to form a (r, cr, (p1)^2, (p2)^2)-LSH
- For million = 2^20 points, D ~ 256 = 2^8, #bucket = (2D)^2 = 2^18 suffice

- TODO:
For billion 2^30 points, we might need to use 3 concatenating LSH, i.e 2^27 buckets
**/
void scaledFalconnCEOsIndexing2_iProbes_1D()
{
//    cout << "Build Scaled FalconnCEOs structure..." << endl;
    auto start = chrono::high_resolution_clock::now();

    // PARAM_INTERNAL_FWHT_PROJECTION = 2^log{D}
    // There will be some case we use less than D random projection, so call FWHT with 2^log{D} and select top-D-up positions
    // This trick is also presented in Falconn
    bitHD3Generator2(PARAM_INTERNAL_FWHT_PROJECTION * PARAM_LSH_NUM_TABLE * PARAM_INTERNAL_NUM_ROTATION);

    float fScaleData = 0.0;
    int iLowerBound_Count = 0;

    // # bucket = (2 * D)^2 since we use 2 combined LSH
    // PARAM_INTERNAL_LSH_NUM_BUCKET = 2 * NUM_PROJECTION
    int NUM_BUCKET = PARAM_INTERNAL_LSH_NUM_BUCKET * PARAM_INTERNAL_LSH_NUM_BUCKET;

    /**
    Data structure:
    - VECTOR_FALCONN_TABLES is an 1D array, containing NUM-BUCKET vector Li,
    each vector L1 contains points sorted by <x, ri> + <x, si> (due to CEOs property)
    where ri and si are random vectors from 2 combined LSH

    - VECTOR_PAIR_FALCONN_BUCKET_POS is a vector of pair where
    pair.first is the beginning of the bucket in VECTOR_FALCONN_TABLES
    pair.second is the size of the bucket (i.e. size of Li)
    **/
    // pair.first is the bucket pos in a big 1D array, which is often large (L * n = 2^10 * 2^20 = 2^30), so uint32_t is okie
    // Note that we might need to use uint64_t in some large data set.
    // However, it demands more space for this array, which might be not cache-efficient
    // pair.second is the bucket size, which is often small, so uint16_t is more than enough

    VECTOR_PAIR_FALCONN_BUCKET_POS = vector<pair<uint32_t, uint16_t>> (PARAM_LSH_NUM_TABLE * NUM_BUCKET);

    // No need to init since it will be updated on-the-fly
    // VECTOR_FALCONN_TABLES = vector<IVector> (PARAM_LSH_NUM_TABLE * NUM_BUCKET);

    #pragma omp parallel for
	for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
	{
        //cout << "Hash Table " << l << endl;
        int iBaseTableIdx = l * NUM_BUCKET;

        // vecBucket_MaxQue is a table, each bucket is a max priQueue
        vector< priority_queue< IFPair, vector<IFPair> > > vecBucket_MaxQue(NUM_BUCKET);

        /**
        Build a hash table for N points
        **/
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            VectorXf rotatedX1 = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION);
            rotatedX1.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);

            VectorXf rotatedX2 = rotatedX1;

            for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
            {
//                rotatedX1 = rotatedX1.cwiseProduct(matHD1.col(r));
//                rotatedX2 = rotatedX2.cwiseProduct(matHD2.col(r));

                for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
                {
                    rotatedX1(d) *= (2 * (int)bitHD1[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                    rotatedX2(d) *= (2 * (int)bitHD2[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                }

                fht_float(rotatedX1.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
                fht_float(rotatedX2.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
            }

            // This queue is used for finding top-k max hash values and hash index
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueProbes1;
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueProbes2;

            /**
            We use a priority queue to keep top-iProbes abs projection for each LSH functions
            **/
            for (int r = 0; r < PARAM_LSH_NUM_PROJECTION; ++r)
            {
                // 1st LSH
                int iSign = sgn(rotatedX1(r));
                float fAbsHashValue = iSign * rotatedX1(r);

                // if iSign >= 0: hash value = {0, ..., r-1}
                // if iSign < 0: hash value = {r, ..., 2r-1}
                int iHashIndex = r;
                if (iSign < 0)
                    iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION; // set bit at position log2(D)

                if ((int)minQueProbes1.size() < PARAM_LSH_NUM_INDEX_PROBES)
                    minQueProbes1.push(IFPair(iHashIndex, fAbsHashValue));

                // in case full queue
                else if (fAbsHashValue > minQueProbes1.top().m_fValue)
                {
                    minQueProbes1.pop();
                    minQueProbes1.push(IFPair(iHashIndex, fAbsHashValue));
                }

                // 2nd LSH
                iSign = sgn(rotatedX2(r));
                fAbsHashValue = iSign * rotatedX2(r);

                // if iSign >= 0: hash value = {0, ..., r-1}
                // if iSign < 0: hash value = {r, ..., 2r-1}
                iHashIndex = r;
                if (iSign < 0)
                    iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION; // set bit at position log2(D)

                if ((int)minQueProbes2.size() < PARAM_LSH_NUM_INDEX_PROBES)
                    minQueProbes2.push(IFPair(iHashIndex, fAbsHashValue));

                // in case full queue
                else if (fAbsHashValue > minQueProbes2.top().m_fValue)
                {
                    minQueProbes2.pop();
                    minQueProbes2.push(IFPair(iHashIndex, fAbsHashValue));
                }
            }

//            assert((int)minQueProbes1.size() == PARAM_LSH_NUM_INDEX_PROBES);
//            assert((int)minQueProbes2.size() == PARAM_LSH_NUM_INDEX_PROBES);

            // Convert to vector
            vector<IFPair> vec1(PARAM_LSH_NUM_INDEX_PROBES);
            vector<IFPair> vec2(PARAM_LSH_NUM_INDEX_PROBES);

            for (int p = PARAM_LSH_NUM_INDEX_PROBES - 1; p >= 0; --p)
            {
                vec1[p] = minQueProbes1.top();
                minQueProbes1.pop();

                vec2[p] = minQueProbes2.top();
                minQueProbes2.pop();
            }

            /**
            Find the top-iProbes over 2 LSH (bucketIndex, abs(projValue1) + abs(projValue2)
            **/
            priority_queue<IFPair, vector<IFPair>, greater<IFPair>> minQue;

            for (const auto& ifPair1: vec1)         //p: probing step
            {
                int iHashIndex1 = ifPair1.m_iIndex; //hash value
                float fAbsHashValue1 = ifPair1.m_fValue; //abs(projection value)

                for (const auto& ifPair2: vec2)         //p: probing step
                {
                    int iHashIndex2 = ifPair2.m_iIndex;
                    float fAbsSumHash = ifPair2.m_fValue + fAbsHashValue1;

                    int iBucketIndex = iHashIndex1 * PARAM_INTERNAL_LSH_NUM_BUCKET + iHashIndex2; // (totally we have 2D * 2D buckets)

                    // new pair for inserting into priQueue
                    // assert(iBucketIndex < NUM_BUCKET);

                    // Push points into the bucket
                    if ((int)minQue.size() < PARAM_LSH_NUM_INDEX_PROBES)
                        minQue.push(IFPair(iBucketIndex, fAbsSumHash));

                    else if (fAbsSumHash > minQue.top().m_fValue)
                    {
                        minQue.pop();
                        minQue.push(IFPair(iBucketIndex, fAbsSumHash));
                    }
                }
            }

            /**
            Insert point (n, absProjectionValue) into a bucket presented as a priority queue
            We will have to extract top-(\alpha * B) points in this bucket later.
            **/

            while (!minQue.empty())
            {
                IFPair ifPair = minQue.top(); // index is bucketID, value is sumAbsHash
                minQue.pop();
                vecBucket_MaxQue[ifPair.m_iIndex].push(IFPair(n, ifPair.m_fValue));
            }
        }

        // Only for debug: count # points in a table
//        int iNumPoint = 0;
//        for (int i = 0; i < NUM_BUCKET; ++i)
//        {
//            iNumPoint += vecBucket_MaxQue[i].size();
//        }
//
//        assert(iNumPoint == PARAM_DATA_N * PARAM_LSH_NUM_INDEX_PROBES * PARAM_LSH_NUM_INDEX_PROBES);

        // Convert priorityQueue to vector
        // Now each bucket is a vector sorted by the hash value (projected value)
        // REQUIREMENT: Largest value are at the front of the vector
        int iNumPoint = 0;
        for (int h = 0; h < NUM_BUCKET; ++h )
        {
            // NOTE: must check empty bucket
            if (vecBucket_MaxQue[h].empty())
                continue;

            int iBucketSize = vecBucket_MaxQue[h].size();
            vector<int> vecBucket(iBucketSize, -1); // hack: use pointIdx = -1 to find the bug if happen

            // Since the queue pop the max value first
            for (int i = 0; i < iBucketSize; ++i )
            {
                vecBucket[i] = vecBucket_MaxQue[h].top().m_iIndex;
                vecBucket_MaxQue[h].pop();
            }

            // We must scale to make sure that the number of points is: alpha * N
            int iLimit = (int)ceil(PARAM_LSH_BUCKET_SIZE_SCALE * iBucketSize / PARAM_LSH_NUM_INDEX_PROBES);

            // If the bucket is small, we do not scale it much, keep at least top-K candidates
            // In practice, it is useful
            // For a fair comparison with threshold LSF or Falconn, this IF need to be deleted
            if (PARAM_INTERNAL_LIMIT_BUCKET && (iLimit < PARAM_MIPS_TOP_K))
            {
                iLimit = min(PARAM_MIPS_TOP_K, iBucketSize);
                iLowerBound_Count++; // counting number of small buckets that are not fully scaled by \alpha
            }

            iNumPoint += iLimit;

            /** Build data structure here
            - must set #pragma omp critical since changing one bucket affects other buckets (2D vector would be safer for threading)

            - VECTOR_PAIR_FALCONN_BUCKET_POS: contains pair<beginning of bucket, # points in the bucket>
            Empty bucket will be stored as well with iLimit = 0
            - VECTOR_FALCONN_TABLES: an 1D array contains non-empty buckets, each bucket contain a vector of point Idx (sorted by abs(projValue))
            **/
            // First: update the position of the bucket
            // Must use uint32_t since we might deal with large L
            // e.g. L = 2^10 * (2^10 * 2^10) = 2^32
            #pragma omp critical
            {
                VECTOR_PAIR_FALCONN_BUCKET_POS[iBaseTableIdx + h] = make_pair((uint32_t)VECTOR_FALCONN_TABLES.size(), (uint16_t)iLimit);

    //            #pragma omp critical
//                int temp = VECTOR_FALCONN_TABLES.size();

                // Second: add the bucket into a 1D vector
                // This is the global data structure, it must be declared as critical
                //#pragma omp critical
                VECTOR_FALCONN_TABLES.insert(VECTOR_FALCONN_TABLES.end(), vecBucket.begin(), vecBucket.begin() + iLimit);

    //            #pragma omp critical
//                assert(VECTOR_FALCONN_TABLES.size() - temp == iLimit);
            }


        }

        fScaleData += (1.0 * iNumPoint / PARAM_DATA_N) / PARAM_LSH_NUM_TABLE;
	}

	//shink_to_fit
	VECTOR_FALCONN_TABLES.shrink_to_fit();
	VECTOR_PAIR_FALCONN_BUCKET_POS.shrink_to_fit();

//	cout << "Finish building index... " << endl;
//    cout << "Size of VECTOR_FALCONN_TABLES using sizeof() in bytes: " << sizeof(VECTOR_FALCONN_TABLES) << endl;
//    cout << "Size of an element in bytes: " << sizeof(VECTOR_FALCONN_TABLES[0]) << endl;
//    cout << "Number of element: " << VECTOR_FALCONN_TABLES.size() << endl;

    double dTemp = 1.0 * sizeof(VECTOR_FALCONN_TABLES)  / (1 << 30) +
                   1.0 * sizeof(VECTOR_FALCONN_TABLES[0]) * VECTOR_FALCONN_TABLES.size() / (1 << 30) ; // capacity() ?

//    cout << "Size of VECTOR_FALCONN_TABLES in GB by sum sizeof() + capacity() * 4: " << dTemp << endl;

    double dIndexSize = dTemp; // convert to GB


//    cout << "Size of VECTOR_PAIR_FALCONN_BUCKET_POS using sizeof() in bytes:  " << sizeof(VECTOR_PAIR_FALCONN_BUCKET_POS) << endl;
//    cout << "Size of element in bytes: " << sizeof(VECTOR_PAIR_FALCONN_BUCKET_POS[0]) << endl;
//    cout << "Number of lement: " << VECTOR_PAIR_FALCONN_BUCKET_POS.size() << endl;

    dTemp = 1.0 * sizeof(VECTOR_PAIR_FALCONN_BUCKET_POS) / (1 << 30) +
            1.0 * sizeof(VECTOR_PAIR_FALCONN_BUCKET_POS[0]) * VECTOR_PAIR_FALCONN_BUCKET_POS.size() / (1 << 30); // in GB

//    cout << "Size of VECTOR_PAIR_FALCONN_BUCKET_POS in GB by sum sizeof() + capacity() * 4: " << dTemp << endl;

    dIndexSize += dTemp;  // convert to GB

    cout << "Size of 1D ScaledFalconnCEOs2 index in GB: " << dIndexSize << endl;

    cout << "numPoints in Table / N: " << fScaleData << endl;
    cout << "Percentage of partially scaled buckets in a table: " << 1.0 * iLowerBound_Count / (NUM_BUCKET * PARAM_LSH_NUM_TABLE) << endl;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Construct 1D ScaledFalconnCEOs2 Data Structure Wall Time (in seconds): " << (float)duration.count() << " seconds" << endl;

}

/**
Theoretical LSF with fixed global threshold to govern the size of index
- Worse performance than Falconn++
**/
void thresFalconnCEOsIndexing2_1D()
{
//    cout << "Build Scaled FalconnCEOs structure..." << endl;
    auto start = chrono::high_resolution_clock::now();

    bitHD3Generator2(PARAM_INTERNAL_FWHT_PROJECTION * PARAM_LSH_NUM_TABLE * PARAM_INTERNAL_NUM_ROTATION);

//    float fThres = PARAM_LSH_BUCKET_SIZE_SCALE * sqrt(2 * log(PARAM_LSH_NUM_PROJECTION)) * PARAM_LSH_NUM_PROJECTION;
//    cout << "Scaled threshold of " << PARAM_LSH_BUCKET_SIZE_SCALE << " is " << fThres << endl;

    // Pr(x in a table) = alpha --> Pr(x in a bucket) = alpha / #buckets
    // Pr(x. r >= t) * Pr(x. r >= t) = alpha / #buckets
    float fThres = 1 - sqrt(PARAM_LSH_BUCKET_SIZE_SCALE) / PARAM_INTERNAL_LSH_NUM_BUCKET;
    fThres = NormalCDFInverse(fThres);
    cout << "Threshold of " << PARAM_LSH_BUCKET_SIZE_SCALE << " is " << fThres << endl;
    fThres *= PARAM_INTERNAL_FWHT_PROJECTION; //
    cout << "Scaled threshold of " << PARAM_LSH_BUCKET_SIZE_SCALE << " is " << fThres << endl;

    float fScaleData = 0.0;

    int NUM_BUCKET = PARAM_INTERNAL_LSH_NUM_BUCKET * PARAM_INTERNAL_LSH_NUM_BUCKET;
    VECTOR_PAIR_FALCONN_BUCKET_POS = vector<pair<uint32_t, uint16_t>> (PARAM_LSH_NUM_TABLE * NUM_BUCKET);

    #pragma omp parallel for
	for (int l = 0 ; l < PARAM_LSH_NUM_TABLE; ++l)
	{
        //cout << "Hash Table " << l << endl;
        int iBaseTableIdx = l * NUM_BUCKET;

        // vecMaxQue is a hash table, each bucket is a priority queue
        vector< vector<IFPair> > vecBuckets(NUM_BUCKET);

        /**
        Build a hash table for N points
        **/
        for (int n = 0; n < PARAM_DATA_N; ++n)
        {
            VectorXf rotatedX1 = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION);
            rotatedX1.segment(0, PARAM_DATA_D) = MATRIX_X.col(n);

            VectorXf rotatedX2 = rotatedX1;

            for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
            {
                for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
                {
                    rotatedX1(d) *= (2 * (int)bitHD1[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                    rotatedX2(d) *= (2 * (int)bitHD2[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                }

                fht_float(rotatedX1.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
                fht_float(rotatedX2.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
            }

//            cout << rotatedX1 << endl;

            vector<IFPair> vec1;
            vector<IFPair> vec2;

            /**
            We use a priority queue to keep top-max abs projection for each repeatation
            **/
            // 1st rotation
            for (int r = 0; r < PARAM_LSH_NUM_PROJECTION; ++r)
            {
                int iSign = sgn(rotatedX1(r));
                float fAbsHashValue = iSign * rotatedX1(r); // work only for numRotate = 3

                // Only keep points that is larger than the threshold
                if (fAbsHashValue >= fThres)
                {
                    int iHashIndex = r;
                    if (iSign < 0)
                        iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION; // set bit at position log2(D)

                    vec1.push_back(IFPair(iHashIndex, fAbsHashValue));
                }

                iSign = sgn(rotatedX2(r));
                fAbsHashValue = iSign * rotatedX2(r); // work only for numRotate = 3

                // Only keep points that is larger than the threshold
                if (fAbsHashValue >= fThres)
                {
                    int iHashIndex = r;
                    if (iSign < 0)
                        iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION; // set bit at position log2(D)

                    vec2.push_back(IFPair(iHashIndex, fAbsHashValue));
                }
            }

//            cout << vec1.size() << " " << vec2.size() << endl;

            // No need sorting
//            sort(vec1.begin(), vec1.end(), greater<IFPair>());
//            sort(vec2.begin(), vec2.end(), greater<IFPair>());

            /**
            Insert point (n, absProjectionValue) into a bucket as a priority queue
            **/
            for (const auto& ifPair1: vec1)         //p: probing step
            {
                int iHashIndex1 = ifPair1.m_iIndex;
                float fAbsHashValue1 = ifPair1.m_fValue;

                for (const auto& ifPair2: vec2)         //p: probing step
                {
                    int iHashIndex2 = ifPair2.m_iIndex;
                    float fAbsHashValue2 = ifPair2.m_fValue;

                    int iBucketIndex = iHashIndex1 * PARAM_INTERNAL_LSH_NUM_BUCKET + iHashIndex2; // (totally we have 2D * 2D buckets)

                    // new pair for inserting into priQueue
//                    assert(iBucketIndex < NUM_BUCKET);

                    // Push all points into the bucket
                    vecBuckets[iBucketIndex].push_back(IFPair(n, fAbsHashValue1 + fAbsHashValue2));
                }
            }

//            assert((int)minQueProbes1.size() == 0);
//            assert((int)minQueProbes2.size() == 0);
        }

//        int iNumPoint = 0;
//        for (int i = 0; i < NUM_BUCKET; ++i)
//        {
//            iNumPoint += vecBucket_MaxQue[i].size();
//        }

//        cout << "Total number of points in a hash table: " << iNumPoint << endl;

//        assert(iNumPoint == PARAM_DATA_N * PARAM_LSH_NUM_INDEX_PROBES * PARAM_LSH_NUM_INDEX_PROBES);

        // Convert priorityQueue to vector
        // Now each bucket is a vector sorted by the hash value (projected value)
        // REQUIREMENT: Largest value are at the front of the vector
        int iNumPoint = 0;
        for (int h = 0; h < NUM_BUCKET; ++h )
        {
            // NOTE: must check empty bucket
            if (vecBuckets[h].empty())
                continue;

            // sort to make sure we can get largest estimate first
            sort(vecBuckets[h].begin(), vecBuckets[h].end(), greater<IFPair>());

            int iBucketSize = vecBuckets[h].size();

            IVector vecBucket(iBucketSize, -1);
            for (int i = 0; i < iBucketSize; ++i)
            {
//                assert(vecBucket_MaxQue[h][i].m_iIndex < PARAM_DATA_N);
//                assert(vecBucket_MaxQue[h][i].m_iIndex >= 0);

                vecBucket[i] = vecBuckets[h][i].m_iIndex; //m_iIndex is now pointIdx
            }


            iNumPoint += iBucketSize;

            #pragma omp critical
            {
                VECTOR_PAIR_FALCONN_BUCKET_POS[iBaseTableIdx + h] = make_pair((uint32_t)VECTOR_FALCONN_TABLES.size(), (uint16_t)iBucketSize);
                VECTOR_FALCONN_TABLES.insert(VECTOR_FALCONN_TABLES.end(), vecBucket.begin(), vecBucket.end());
            }

        }

        fScaleData += (1.0 * iNumPoint / PARAM_DATA_N) / PARAM_LSH_NUM_TABLE;
	}

//	cout << "Finish building index... " << endl;

    cout << "Avg of numPoints in Table / N = " << fScaleData << endl;
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Construct 1D thresFalconnCEOs2_Pair Data Structure Wall Time (in seconds): " << (float)duration.count() << " seconds" << endl;

}

