#include <chrono>
#include <iostream>
#include <queue>
#include <fstream>
#include <string>

#include "FindKNN.h"
#include "Header.h"
#include "Utilities.h"

// #include "space_ip.h" // Sometime bug appears with AVX512

/**
There will be two ways to probe:
- Sequencial probing where we use qProbes/L for each table (might be faster in querying but lower recall)
- Adaptive probing where we consider the best buckets (ranking based on abs(projValue1) + abs(projValue2) among ALL tables
**/
void FalconnCEOs2_1D_TopK()
{
//    cout << "Scaled FalconnCEOs Cyclic Probes querying..." << endl;

    auto startTime = chrono::high_resolution_clock::now();

    // useful for debug
//    float hashTime = 0, lookupTime = 0, distTime = 0;
//	uint64_t iTotalProbes = 0, iTotalUniqueCand = 0, iTotalCand = 0;
//    int iNumEmptyBucket = 0;

    MatrixXi matTopK = MatrixXi::Zero(PARAM_MIPS_TOP_K, PARAM_QUERY_Q);

    int NUM_BUCKET = PARAM_INTERNAL_LSH_NUM_BUCKET * PARAM_INTERNAL_LSH_NUM_BUCKET;


    // Trick: Only sort to get top-maxProbe since we do not need the rest.
    // This will reduce the cost of LDlogD to LDlog(maxProbe) for faster querying
    // Otherwise, we have to use on-the-fly sorting (similar to Falconn)
    // 2.0* should have enough number of probes per 1 LSH function to extract the top-k projection values
    int iMaxProbesPerTable = ceil(2.0 * PARAM_LSH_NUM_QUERY_PROBES / PARAM_LSH_NUM_TABLE);
    int iMaxProbesPerHash = ceil(sqrt(1.0 * iMaxProbesPerTable)); // since we combine 2 LSH functions

//    cout << "Max probes per table is " << iMaxProbesPerTable << endl;
//    cout << "Max probes per rotation is " << iMaxProbesPerRotate << endl;

//    int num_threads = 64;
//    omp_set_num_threads(num_threads); //set the number of threads
//
//    #pragma omp parallel for reduction(+:hashTime, lookupTime, distTime, iTotalProbes, iTotalUniqueCand, iTotalCand) //num_threads(num_threads)
    #pragma omp parallel for
	for (int q = 0; q < PARAM_QUERY_Q; ++q)
	{
//		auto startTime = chrono::high_resolution_clock::now();

		// Get hash value of all hash table first
		VectorXf vecQuery = MATRIX_Q.col(q);

		// For each hash table, keep top-m largest projections for query probing
		// We use a priority queue to keep track the projection value
		vector<priority_queue< IFPair, vector<IFPair>, greater<IFPair> >> vecMinQue(PARAM_LSH_NUM_TABLE);

		/** Rotating and prepared probes sequence **/
		for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            VectorXf rotatedQ1 = VectorXf::Zero(PARAM_INTERNAL_FWHT_PROJECTION);
            rotatedQ1.segment(0, PARAM_DATA_D) = vecQuery;

            VectorXf rotatedQ2 = rotatedQ1;

            for (int r = 0; r < PARAM_INTERNAL_NUM_ROTATION; ++r)
            {

                for (int d = 0; d < PARAM_INTERNAL_FWHT_PROJECTION; ++d)
                {
                    rotatedQ1(d) *= (2 * (int)bitHD1[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                    rotatedQ2(d) *= (2 * (int)bitHD2[l * PARAM_INTERNAL_NUM_ROTATION * PARAM_LSH_NUM_PROJECTION + r * PARAM_INTERNAL_FWHT_PROJECTION + d] - 1);
                }

                fht_float(rotatedQ1.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
                fht_float(rotatedQ2.data(), PARAM_INTERNAL_LOG2_FWHT_PROJECTION);
            }


            // Assign hashIndex and compute distance between hashValue and the maxValue
            // Then insert into priority queue
            // Get top-k max position on each rotations
            // minQueue might be better regarding space usage, hence better for cache
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQue1;
            priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQue2;

            for (int r = 0; r < PARAM_LSH_NUM_PROJECTION; ++r)
            {
                // 1st LSH
                int iSign = sgn(rotatedQ1(r));
                float fHashValue = iSign * rotatedQ1(r);

                // Get hashIndex
                int iHashIndex = r;
                if (iSign < 0)
                    iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION;

                // cout << "Hash index 1 : " << iBucketIndex << endl;

                // Trick: We only keep top-iMaxProbesPerHash since
                // Falconn uses block sorting to save sorting time
                if ((int)minQue1.size() < iMaxProbesPerHash)
                    minQue1.push(IFPair(iHashIndex, fHashValue));

                // queue is full
                else if (fHashValue > minQue1.top().m_fValue)
                {
                    minQue1.pop(); // pop max, and push min hash distance
                    minQue1.push(IFPair(iHashIndex, fHashValue));
                }

                // 2nd LSH
                iSign = sgn(rotatedQ2(r));
                fHashValue = iSign * rotatedQ2(r);

                // Get hashIndex
                iHashIndex = r;
                if (iSign < 0)
                    iHashIndex |= 1UL << PARAM_INTERNAL_LOG2_NUM_PROJECTION;

                // cout << "Hash index 2: " << iBucketIndex << endl;

                // hard code on iMaxProbes = 2 * averageProbes: we only keep 2 * averageProbes smallest value
                // Falconn uses block sorting to save sorting time
                if ((int)minQue2.size() < iMaxProbesPerHash)
                    minQue2.push(IFPair(iHashIndex, fHashValue));

                // queue is full
                else if (fHashValue > minQue2.top().m_fValue)
                {
                    minQue2.pop(); // pop max, and push min hash distance
                    minQue2.push(IFPair(iHashIndex, fHashValue));
                }
            }

//            assert((int)minQue1.size() == iMaxProbesPerRotate);
//            assert((int)minQue2.size() == iMaxProbesPerRotate);

            // Convert to vector, the large projection value is in [0]
            // Hence better for creating a sequence of probing since we do not have to call pop() many times
            vector<IFPair> vec1(iMaxProbesPerHash), vec2(iMaxProbesPerHash);
            for (int p = iMaxProbesPerHash - 1; p >= 0; --p)
            {
                // 1st LSH
                IFPair ifPair = minQue1.top();
                minQue1.pop();
                vec1[p] = ifPair;

                // 2nd LSH
                ifPair = minQue2.top();
                minQue2.pop();
                vec2[p] = ifPair;
            }

            // Now begin building the query probes on ONE table
            for (const auto& ifPair1: vec1)
            {
                int iHashIndex1 = ifPair1.m_iIndex;
                float fAbsHashValue1 = ifPair1.m_fValue;

                //cout << "Hash index 1: " << iBucketIndex1 << " projection value: " << fAbsHashValue1 << endl;

                for (const auto& ifPair2: vec2)         //p: probing step
                {
                    int iHashIndex2 = ifPair2.m_iIndex;
                    float fAbsHashValue2 = ifPair2.m_fValue;

                    //cout << "Hash index 2: " << iBucketIndex2 << " projection value: " << fAbsHashValue2 << endl;

                    // Start building the probe sequence
                    int iBucketIndex = iHashIndex1 * PARAM_INTERNAL_LSH_NUM_BUCKET + iHashIndex2; // (totally we have 2D * 2D buckets)
                    float fSumHashValue = fAbsHashValue1 + fAbsHashValue2;

                    assert(iBucketIndex < NUM_BUCKET);

                    // IMPORTANT: Must use ALL iMaxProbesPerTable < iMaxProbesPerRotate^2
                    // since the minQueue will pop the min projection value first
                    // If do not use iMaxProbesPerRotate^2, we miss the bucket of query (max + max)
                    if ((int)vecMinQue[l].size() < iMaxProbesPerTable)
                        vecMinQue[l].push(IFPair(iBucketIndex, fSumHashValue));

                    else if (fSumHashValue > vecMinQue[l].top().m_fValue)
                    {
                        vecMinQue[l].pop(); // pop max, and push min hash distance
                        vecMinQue[l].push(IFPair(iBucketIndex, fSumHashValue));
                    }
                }
            }

//            assert((int)vecMinQue[l].size() == iMaxProbesPerTable);

        }


        // We need to dequeue to get the bucket on the right order
        // Every table has iMaxProbes positions for query probing
        // TODO: use Boost.MultiArray for less maintaining cost, and perhap cache-friendly
        vector<IFPair> vecBucketProbes(PARAM_LSH_NUM_TABLE * iMaxProbesPerTable);
        for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            int iBaseTableIdx = l * NUM_BUCKET;
            int iBaseIdx = l * iMaxProbesPerTable; // this is for vecProbes

            int idx = iMaxProbesPerTable - 1;

            while (!vecMinQue[l].empty())
            {
                // m_iIndex = bucketIndex, mfValue = absHashValue1 + absHashValue2
                IFPair ifPair = vecMinQue[l].top();
                vecMinQue[l].pop();

                //cout << "Bucket index: " << ifPair.m_iIndex << endl;

                // Now: ifPair.m_iIndex is the bucket index
                // changing the index to have TableIdx information since we iterate probing through all tables
                // This index is used to access the hash table VECTOR_FALCONN_TABLES
                ifPair.m_iIndex = iBaseTableIdx + ifPair.m_iIndex;
                // Now: ifPair.m_iIndex contains the position of the table idx & hashIndex

                // Since we pop from minQueue, it need to be stored at the end first (i.e. idx = iMaxProbesPerTable - 1)
                vecBucketProbes[iBaseIdx + idx] = ifPair;
                idx--;
            }

//            printVector(vecProbes[l]);

        }


		// Then preparing multi-probe
        // We get the bucket of query in all tables first
        priority_queue< IFPair, vector<IFPair> > maxQueProbes;
        for (int l = 0; l < PARAM_LSH_NUM_TABLE; ++l)
        {
            // position of query's buckets
            // Query probe criteria: abs(<q, ri>) + abs(<q, si>)
            // The closer q and (ri, si), the better chance to get kNN
            maxQueProbes.push(vecBucketProbes[l * iMaxProbesPerTable]);
        }

//        auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
//        hashTime += (float)durTime.count();

        /** Query probing **/
//        startTime = chrono::high_resolution_clock::now();

		boost::dynamic_bitset<> bitsetHist(PARAM_DATA_N); // all 0's by default
        VectorXi vecProbeTracking = VectorXi::Zero(PARAM_LSH_NUM_TABLE);

        int iNumCand = 0;
        for (int iProbeCount = 0; iProbeCount < PARAM_LSH_NUM_QUERY_PROBES; iProbeCount++)
        {

//            iTotalProbes++;

            // maxQueProbes has all query's buckets of L tables, and they will be used first
            // After using a query's bucket of table T, we insert into maxQueProbes the next bucket of T.
            // The maxQueProbes will determine the next bucket of the hash table to use.
            IFPair ifPair = maxQueProbes.top();
            maxQueProbes.pop();
//            cout << "Probe " << iProbeCount << ": " << ifPair.m_iIndex << " " << ifPair.m_fValue << endl;

            // ifPair.m_iIndex is the (tableIdx & bucketIdx)
            uint32_t iBucketPos = VECTOR_PAIR_FALCONN_BUCKET_POS[ifPair.m_iIndex].first;
            uint16_t iBucketSize = VECTOR_PAIR_FALCONN_BUCKET_POS[ifPair.m_iIndex].second;

            // Update probe tracking
            int iTableIdx = ifPair.m_iIndex / NUM_BUCKET; // get table idx

            //cout << "Table: " << iTableIdx << " Hash index: " << ifPair.m_iIndex - iTableIdx * NUM_BUCKET << endl;
            //printVector(vecBucket);

            vecProbeTracking(iTableIdx)++;

            // insert into the queue for next probes
            if (vecProbeTracking(iTableIdx) < iMaxProbesPerTable)
            {
                // vecBucketProbes has range l * iMaxProbesPerTable + idx (ie top-probes)
                IFPair ifPair = vecBucketProbes[iTableIdx * iMaxProbesPerTable + vecProbeTracking(iTableIdx)]; // get the next bucket idx of the investigated hash table

                // Query probing criteria: abs(<q, ri>) + abs(<q, si>)
                // The closer q and (ri, si), the better chance to get NN
                // It is identical to the CEOs property to get top-s0 min & max vectors
                maxQueProbes.push(ifPair);
            }

            if (iBucketSize == 0)
            {
//                iNumEmptyBucket++;
                continue;
            }

    /**
	For debugging
	**/
//            iTotalCand += iBucketSize;

            // Get all points in the bucket
            for (int i = 0; i < iBucketSize; ++i)
            {
                int iPointIdx = VECTOR_FALCONN_TABLES[iBucketPos + i];

//                assert (iPointIdx < PARAM_DATA_N);
//                assert (iPointIdx >= 0);

                if (~bitsetHist[iPointIdx])
                {
                    iNumCand++;
                    bitsetHist[iPointIdx] = 1;
                }

            }

            // Allow all query probes before reaching the limit
//            if (iNumCand >= PARAM_MIPS_CANDIDATE_SIZE)
//                break;
		}

    /**
	For debugging
	**/
//        durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
//        lookupTime += (float)durTime.count();

//        iTotalProbes += iProbeCount;
//		iTotalUniqueCand += iNumCand; // bitsetHist.count();

//		startTime = chrono::high_resolution_clock::now();

		// in case some query does not have candidate or qProbes is too small
        if (iNumCand == 0)
            continue;

		// This is to get top-K
		priority_queue< IFPair, vector<IFPair>, greater<IFPair> > minQueTopK;

        //new A class to do inner product by SIMD from HnswLib
//        hnswlib::InnerProductSpace inner(PARAM_DATA_D);
//        hnswlib::DISTFUNC<float> fstdistfunc_ = inner.get_dist_func();
//        void *dist_func_param_;
//        dist_func_param_ = inner.get_dist_func_param();


        size_t iPointIdx = bitsetHist.find_first();
        while (iPointIdx != boost::dynamic_bitset<>::npos)
        {
            // Get dot product
//            float fInnerProduct = fstdistfunc_(vecQuery.data(), MATRIX_X.col(iPointIdx).data(), dist_func_param_);
            float fInnerProduct = vecQuery.dot(MATRIX_X.col(iPointIdx));

            // Add into priority queue
            if (int(minQueTopK.size()) < PARAM_MIPS_TOP_K)
                minQueTopK.push(IFPair(iPointIdx, fInnerProduct));

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.push(IFPair(iPointIdx, fInnerProduct));
            }

            iPointIdx = bitsetHist.find_next(iPointIdx);
        }

//        cout << "Queue TopK size: " << minQueTopK.size() << endl;

        // assert((int)minQueTopK.size() == PARAM_MIPS_TOP_K);

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
//            cout << "Bug is here at " << k << endl;

            matTopK(k, q) = minQueTopK.top().m_iIndex;
            minQueTopK.pop();
        }

    /**
	For debugging
	**/
//		durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
//		distTime += (float)durTime.count();
	}

	auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);

	/**
	For debugging
	**/
//	cout << "Finish querying..." << endl;
//    cout << "Average number of empty buckets per query: " << (double)iNumEmptyBucket / PARAM_QUERY_Q << endl;
//	cout << "Average number of probes per query: " << (double)iTotalProbes / PARAM_QUERY_Q << endl;
//	cout << "Average number of unique candidates per query: " << (double)iTotalUniqueCand / PARAM_QUERY_Q << endl;
//	cout << "Average number of candidates per query: " << (double)iTotalCand / PARAM_QUERY_Q << endl;

//	cout << "Hash and Probing Time: " << hashTime << " ms" << endl;
//	cout << "Lookup Time: " << lookupTime << " ms" << endl;
//	cout << "Distance Time: " << distTime << " ms" << endl;

	cout << "Querying Time: " << (float)durTime.count() << " ms" << endl;


	if (PARAM_INTERNAL_SAVE_OUTPUT)
	{
        string sFileName = PARAM_OUTPUT_FILE + "_TopK_" + int2str(PARAM_MIPS_TOP_K) +
                        "_NumProjection_" + int2str(PARAM_LSH_NUM_PROJECTION) +
                        "_NumTable_" + int2str(PARAM_LSH_NUM_TABLE) +
                        "_IndexProbe_"  + int2str(PARAM_LSH_NUM_INDEX_PROBES) +
                        "_QueryProbe_"  + int2str(PARAM_LSH_NUM_QUERY_PROBES) +
                        "_BucketScale_"  + int2str((int)(PARAM_LSH_BUCKET_SIZE_SCALE * 100)) +
                        "_CandidateSize_" + int2str(PARAM_MIPS_CANDIDATE_SIZE) + ".txt";

        outputFile(matTopK, sFileName);
	}

}

