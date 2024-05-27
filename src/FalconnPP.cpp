//
// Created by npha145 on 15/03/24.
//

#include "FalconnPP.h"
#include "Header.h"
#include "Utilities.h"
// __builtin_popcount function
// #include <bits/stdc++.h>

/**
 * Used on NeurIPS 2022.
 * We build 2 layers LSH using 1D array
 * https://stackoverflow.com/questions/17259877/1d-or-2d-array-whats-faster
 * Organize L tables, each with 4D^2 buckets as 1D vector for memory efficiency for static dataset
 * As there are many empty buckets, we save significantly memory compared to the nested 2D vector
 * However it might be slow as not parallel-friendly.
 * If indexing require updates, then 1D array is very slow.
 */

void FalconnPP::build2Layers_1D(const Ref<const MatrixXf> & matX){

    cout << "n_points: " << FalconnPP::n_points << endl;
    cout << "n_features: " << FalconnPP::n_features << endl;
    cout << "n_tables: " << FalconnPP::n_tables << endl;
    cout << "n_proj: " << FalconnPP::n_proj << endl;
    cout << "fhtDim: " << FalconnPP::fhtDim << endl;
    cout << "# iProbes: " << FalconnPP::iProbes << endl;
    cout << "bucket_minSize: " << FalconnPP::bucket_minSize << endl;
    cout << "bucket_scale: " << FalconnPP::bucket_scale << endl;

    // We need to center the data set, note that matrix_X is col-wise (D x N)
    VectorXf vecCenter = matX.rowwise().mean();
    FalconnPP::matrix_X = matX.array().colwise() - vecCenter.array(); // must add colwise()

     auto start = chrono::high_resolution_clock::now();

     // Since we have 2 layers, we call HD3Generator2() to generate 2 bitHD, each for each layer
     // We have L tables, each require n_rotate = 3 random rotations.
     // Each random rotation require one random sign vectors of length fhtDim
     FalconnPP::bitHD3Generator2(FalconnPP::fhtDim * FalconnPP::n_tables * FalconnPP::n_rotate);

     double dScaleData = 0.0;
     int iNumLimitedBucket = 0;

     int log2_D = log2(FalconnPP::n_proj);
     int log2_FWHT_D = log2(FalconnPP::fhtDim);

     // # bucket = (2D)^2 as each layer has 2D buckets
     // # bucket should be ~ n_points
     int numBucketsPerTable = 4 * FalconnPP::n_proj * FalconnPP::n_proj;


     // pair.first is the bucket pos in a big 1D array, which is often large
     // (L * n = 2^10 * 2^20 = 2^30), so uint32_t is okie for million-point datasets
     // Note that we might need to use uint64_t in some larger data set.
     // However, it demands more space for this array, which is not cache-efficient
     // pair.second is the bucket size, which is often small, so uint16_t is more than enough
     FalconnPP::vecPair_BucketPos = vector<pair<uint32_t, uint16_t>> (FalconnPP::n_tables * numBucketsPerTable);

     // omp_set_dynamic(0);     // Explicitly disable dynamic teams
     omp_set_num_threads(FalconnPP::n_threads);
#pragma omp parallel for
     for (int l = 0 ; l < FalconnPP::n_tables; ++l)
     {
         //cout << "Hash Table " << l << endl;
         int iBaseTableIdx = l * numBucketsPerTable;

         // vecMaxQue is a hash table, each element is a bucket as a priority queue, some bucket might be empty
         vector< priority_queue< IFPair, vector<IFPair> > > vecBucket_MaxQue(numBucketsPerTable);

         /**
         Build a hash table for N points
         **/
         for (int n = 0; n < FalconnPP::n_points; ++n)
         {
             VectorXf rotatedX1 = VectorXf::Zero(FalconnPP::fhtDim);
             rotatedX1.segment(0, FalconnPP::n_features) = FalconnPP::matrix_X.col(n);

             VectorXf rotatedX2 = rotatedX1;

             for (int r = 0; r < FalconnPP::n_rotate; ++r)
             {
                 // Multiply with random sign
                 for (int d = 0; d < FalconnPP::fhtDim; ++d)
                 {
                     rotatedX1(d) *= (2 * static_cast<float>(FalconnPP::bitHD1[l * FalconnPP::n_rotate * FalconnPP::fhtDim + r * FalconnPP::fhtDim + d]) - 1);
                     rotatedX2(d) *= (2 * static_cast<float>(FalconnPP::bitHD2[l * FalconnPP::n_rotate * FalconnPP::fhtDim + r * FalconnPP::fhtDim + d]) - 1);
                 }

                 fht_float(rotatedX1.data(), log2_FWHT_D);
                 fht_float(rotatedX2.data(), log2_FWHT_D);
             }

             // This queue is used for finding top-k max hash values and hash index for iProbes on each layer
             priority_queue< IFPair, vector<IFPair>, greater<> > minQueProbes1; // 1st layer
             priority_queue< IFPair, vector<IFPair>, greater<> > minQueProbes2; // 2nd layer

             /**
             We use a priority queue to keep top-max abs projection for each repeat
             Always ensure fhtDim >= n_proj
             **/
             for (int r = 0; r < FalconnPP::n_proj; ++r)
             {
                 // 1st rotation
                 int iSign = sgn(rotatedX1(r));
                 float fAbsHashValue = iSign * rotatedX1(r);

                 int iBucketIndex = r;
                 if (iSign < 0)
                     iBucketIndex |= 1UL << log2_D; // set bit at position log2(D)

                 if ((int)minQueProbes1.size() < FalconnPP::iProbes)
                     minQueProbes1.emplace(iBucketIndex, fAbsHashValue); // emplace is push withou creating temp data

                 // in case full queue
                 else if (fAbsHashValue > minQueProbes1.top().m_fValue)
                 {
                     minQueProbes1.pop();
                     minQueProbes1.emplace(iBucketIndex, fAbsHashValue); // No need IFPair
                 }

                 // 2nd rotation
                 iSign = sgn(rotatedX2(r));
                 fAbsHashValue = iSign * rotatedX2(r);

                 iBucketIndex = r;
                 if (iSign < 0)
                     iBucketIndex |= 1UL << log2_D; // set bit at position log2(D)

                 if ((int)minQueProbes2.size() < FalconnPP::iProbes)
                     minQueProbes2.emplace(iBucketIndex, fAbsHashValue);

                     // in case full queue
                 else if (fAbsHashValue > minQueProbes2.top().m_fValue)
                 {
                     minQueProbes2.pop();
                     minQueProbes2.emplace(iBucketIndex, fAbsHashValue);
                 }
             }

    //            assert((int)minQueProbes1.size() == FalconnPP::iProbes);
    //            assert((int)minQueProbes2.size() == FalconnPP::iProbes);

             // Convert to vector
             vector<IFPair> vec1(FalconnPP::iProbes);
             vector<IFPair> vec2(FalconnPP::iProbes);

             for (int p = FalconnPP::iProbes - 1; p >= 0; --p)
             {
                 vec1[p] = minQueProbes1.top();
                 minQueProbes1.pop();

                 vec2[p] = minQueProbes2.top();
                 minQueProbes2.pop();
             }

             /**
             Use minQue to find the top-iProbes over 2 layers via sum of 2 estimators
             Note that vec1 and vec2 are already sorted, and has length of iProbes
             **/
             priority_queue<IFPair, vector<IFPair>, greater<>> minQue;

             for (const auto& ifPair1: vec1)         //p: probing step
             {
                 int iBucketIndex1 = ifPair1.m_iIndex;
                 float fAbsHashValue1 = ifPair1.m_fValue;

                 for (const auto& ifPair2: vec2)         //p: probing step
                 {
                     int iBucketIndex2 = ifPair2.m_iIndex;
                     float fAbsSumHash = ifPair2.m_fValue + fAbsHashValue1; // sum of 2 estimators

                     int iBucketIndex = iBucketIndex1 * (2 * FalconnPP::n_proj) + iBucketIndex2; // (totally we have 2D * 2D buckets)

                     // new pair for inserting into priQueue
                     // assert(iBucketIndex < NUM_BUCKET);

                     // Push all points into the bucket
                     if ((int)minQue.size() < FalconnPP::iProbes)
                         minQue.emplace(iBucketIndex, fAbsSumHash);

                     else if (fAbsSumHash > minQue.top().m_fValue)
                     {
                         minQue.pop();
                         minQue.emplace(iBucketIndex, fAbsSumHash);
                     }
                 }
             }

             /**
             Insert point (n, absProjectionValue) into a bucket as a priority queue
             We will have to extract top-percentage points in this queue later.
             **/

             while (!minQue.empty())
             {
                 IFPair ifPair = minQue.top(); // index is bucketID, value is sumAbsHash
                 minQue.pop();
                 vecBucket_MaxQue[ifPair.m_iIndex].emplace(n, ifPair.m_fValue);
             }
         }

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
         for (int h = 0; h < numBucketsPerTable; ++h )
         {
             // NOTE: must check empty bucket
             if (vecBucket_MaxQue[h].empty())
                 continue;

             int iBucketSize = vecBucket_MaxQue[h].size();
             vector<int> vecBucket(iBucketSize, -1); // hack: use -1 to find the bug if happen

             // Since the queue pop the max value first
             for (int i = 0; i < iBucketSize; ++i )
             {
                 vecBucket[i] = vecBucket_MaxQue[h].top().m_iIndex;
                 vecBucket_MaxQue[h].pop();
             }

             // We must scale to make sure that the number of points is: scale * N
             int iLimit = (int)ceil(FalconnPP::bucket_scale * iBucketSize / FalconnPP::iProbes);

             // For small scaled bucket (sparse area), we scale down to bucket_minSize
             // In practice, it is helpful when queries fall into sparse areas though we cannot guarantee # points in a table
             // For fair comparison with threshold LSF, we must scale the bucket
             if (iLimit < FalconnPP::bucket_minSize)
             {
                 iLimit = min(FalconnPP::bucket_minSize, iBucketSize);
                 iNumLimitedBucket++;
             }

             iNumPoint += iLimit;

             // Change the way to index here
             // First: update the position of the bucket
             // Must use uint32_t since we might deal with large L
             // e.g. L = 2^10 * (2^10 * 2^10) = 2^32
    #pragma omp critical
             {
                 // Store the first and last positions for each bucket on each table here
                 FalconnPP::vecPair_BucketPos[iBaseTableIdx + h] = make_pair((uint32_t)FalconnPP::vecTables_1D.size(), (uint16_t)iLimit);

                 //            #pragma omp critical
    //                int temp = FalconnPP::vecTables.size();

                 // Second: add the bucket into a 1D vector
                 // Push the new bucket at the end of the vecTables
                 //#pragma omp critical
                 FalconnPP::vecTables_1D.insert(FalconnPP::vecTables_1D.end(), vecBucket.begin(), vecBucket.begin() + iLimit);

                 //            #pragma omp critical
    //                assert(FalconnPP::vecTables.size() - temp == iLimit);
             }
         }

         dScaleData += (1.0 * iNumPoint / FalconnPP::n_points) / FalconnPP::n_tables;
     }

     //shink_to_fit
     FalconnPP::vecTables_1D.shrink_to_fit();
     FalconnPP::vecPair_BucketPos.shrink_to_fit();

    cout << "Finish building index... " << endl;

    //    cout << "Size of FalconnPP::vecTables using sizeof() in bytes: " << sizeof(VECTOR_FALCONN_TABLES) << endl;
    //    cout << "Size of an element in bytes: " << sizeof(FalconnPP::vecTables[0]) << endl;
    //    cout << "Number of element: " << FalconnPP::vecTables.size() << endl;

     double dSize_1D = 1.0 * sizeof(FalconnPP::vecTables_1D)  / (1 << 30) +
                    1.0 * sizeof(FalconnPP::vecTables_1D[0]) * FalconnPP::vecTables_1D.size() / (1 << 30) ; // capacity() ?

     cout << "Size of vecTables_1D in GB: " << dSize_1D << endl;

    //    cout << "Size of FalconnPP::vecPair_BucketPos using sizeof() in bytes:  " << sizeof(VECTOR_PAIR_FALCONN_BUCKET_POS) << endl;
    //    cout << "Size of elements in bytes: " << sizeof(FalconnPP::vecPair_BucketPos[0]) << endl;
    //    cout << "Number of elements: " << FalconnPP::vecPair_BucketPos.size() << endl;

     double dSize_Pos = 1.0 * sizeof(FalconnPP::vecPair_BucketPos) / (1 << 30) +
             1.0 * sizeof(FalconnPP::vecPair_BucketPos[0]) * FalconnPP::vecPair_BucketPos.size() / (1 << 30); // in GB

     cout << "Size of vecPair_BucketPos in GB: " << dSize_Pos << endl;

     cout << "Size of Falconn++ (2 layers) index in GB: " << dSize_1D + dSize_Pos << endl;
     cout << "n_points per table / n: " << dScaleData << endl;
     cout << "Percentage of scaled buckets in a table: " << 1.0 * iNumLimitedBucket / (numBucketsPerTable * FalconnPP::n_tables) << endl;

     auto end = chrono::high_resolution_clock::now();
     auto duration = chrono::duration_cast<chrono::seconds>(end - start);
     cout << "Construct Falconn++ (2 layers) Wall Time (in seconds): " << (float)duration.count() << " seconds" << endl;

}

/**
* Used on NeurIPS 2022
* Query on 2 layers LSH
* Adaptively select better buckets among 2D*2D buckets to have better candidate, given the same candSize
*/
MatrixXi FalconnPP::query2Layers_1D(const Ref<const MatrixXf> & matQ, int n_neighbors, bool verbose){

    int n_queries = matQ.cols();

    if (verbose)
    {
        cout << "number of queries: " << n_queries << endl;
        cout << "# qProbes: " << FalconnPP::qProbes << endl;
        cout << "number of threads: " << FalconnPP::n_threads << endl;
    }

     auto startQueryTime = chrono::high_resolution_clock::now();

     float hashTime = 0, lookupTime = 0, distTime = 0;
     uint64_t iTotalProbes = 0, iTotalUniqueCand = 0, iTotalCand = 0;

     MatrixXi matTopK = MatrixXi::Zero(n_neighbors, n_queries);

     int numBucketsPerTable = 4 * FalconnPP::n_proj * FalconnPP::n_proj;
     int iNumEmptyBucket = 0;
     int log2_D = log2(FalconnPP::n_proj);
     int log2_FWHT_D = log2(FalconnPP::fhtDim);

     // Trick: Only sort to get top-maxProbe since we do not need the rest.
     // This will reduce the cost of LDlogD to LDlog(maxProbe) for faster querying
     // 2.0 * should have enough number of probes per rotation to extract the top-k projection values
     int iMaxProbesPerTable = ceil(2.0 * FalconnPP::qProbes / FalconnPP::n_tables);
     int iMaxProbesPerLayer = ceil(sqrt(1.0 * iMaxProbesPerTable)); // one layer

    //    cout << "Max probes per table is " << iMaxProbesPerTable << endl;
    //    cout << "Max probes per rotation is " << iMaxProbesPerLayer << endl;

     // omp_set_dynamic(0);     // Explicitly disable dynamic teams
     omp_set_num_threads(FalconnPP::n_threads);
#pragma omp parallel for reduction(+:hashTime, lookupTime, distTime, iTotalProbes, iTotalUniqueCand, iTotalCand)
     for (int q = 0; q < n_queries; ++q)
     {
         auto startTime = chrono::high_resolution_clock::now();

         // Get hash value of all hash table first
         VectorXf vecQuery = matQ.col(q);

         // For each table, we store the top-m largest projections of |<q, r_i> + <q, s_j>|
         // The index (i, j) --> i * (2D) + j as each layer has 2D buckets. This index will be used as the probing sequence among L tables
         // Therefore, we must keep track these (i, j) pairs on each table
         vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vecMinQue(FalconnPP::n_tables);

         /** Rotating and prepared probes sequence **/
         for (int l = 0; l < FalconnPP::n_tables; ++l)
         {
             VectorXf rotatedQ1 = VectorXf::Zero(FalconnPP::fhtDim);
             rotatedQ1.segment(0, FalconnPP::n_features) = vecQuery;

             VectorXf rotatedQ2 = rotatedQ1;

             for (int r = 0; r < FalconnPP::n_rotate; ++r)
             {

                 for (int d = 0; d < FalconnPP::fhtDim; ++d)
                 {
                     rotatedQ1(d) *= (2 * static_cast<float>(FalconnPP::bitHD1[l * FalconnPP::n_rotate * FalconnPP::fhtDim + r * FalconnPP::fhtDim + d]) - 1);
                     rotatedQ2(d) *= (2 * static_cast<float>(FalconnPP::bitHD2[l * FalconnPP::n_rotate * FalconnPP::fhtDim + r * FalconnPP::fhtDim + d]) - 1);
                 }

                 fht_float(rotatedQ1.data(), log2_FWHT_D);
                 fht_float(rotatedQ2.data(), log2_FWHT_D);
             }


             // Assign hashIndex and compute distance between hashValue and the maxValue
             // Then insert into priority queue
             // Get top-k max position on each rotations
             // minQueue might be better regarding space usage, hence better for cache
             priority_queue< IFPair, vector<IFPair>, greater<> > minQue1;
             priority_queue< IFPair, vector<IFPair>, greater<> > minQue2;

             for (int r = 0; r < FalconnPP::n_proj; ++r)
             {
                 // 1st rotation
                 int iSign = sgn(rotatedQ1(r));
                 float fHashDiff = iSign * rotatedQ1(r);

                 // Get hashIndex
                 int iBucketIndex = r;
                 if (iSign < 0)
                     iBucketIndex |= 1UL << log2_D;

                 // cout << "Hash index 1 : " << iBucketIndex << endl;

                 // hard code on iMaxProbes = 2 * averageProbes: we only keep 2 * averageProbes smallest value
                 // FalconnPP uses block sorting to save sorting time
                 if ((int)minQue1.size() < iMaxProbesPerLayer)
                     minQue1.emplace(iBucketIndex, fHashDiff);

                 // queue is full
                 else if (fHashDiff > minQue1.top().m_fValue)
                 {
                     minQue1.pop(); // pop max, and push min hash distance
                     minQue1.emplace(iBucketIndex, fHashDiff);
                 }

                 // 2nd rotation
                 iSign = sgn(rotatedQ2(r));
                 fHashDiff = iSign * rotatedQ2(r);

                 // Get hashIndex
                 iBucketIndex = r;
                 if (iSign < 0)
                     iBucketIndex |= 1UL << log2_D;

                 // cout << "Hash index 2: " << iBucketIndex << endl;

                 // hard code on iMaxProbes = 2 * averageProbes: we only keep 2 * averageProbes smallest value
                 // FalconnPP uses block sorting to save sorting time
                 if ((int)minQue2.size() < iMaxProbesPerLayer)
                     minQue2.emplace(iBucketIndex, fHashDiff);

                 // queue is full
                 else if (fHashDiff > minQue2.top().m_fValue)
                 {
                     minQue2.pop(); // pop max, and push min hash distance
                     minQue2.emplace(iBucketIndex, fHashDiff);
                 }
             }

    //            assert((int)minQue1.size() == iMaxProbesPerLayer);
    //            assert((int)minQue2.size() == iMaxProbesPerLayer);

             // Convert to vector, the large projection value is in [0]
             // Hence better for creating a sequence of probing since we do not have to call pop() many times
             vector<IFPair> vec1(iMaxProbesPerLayer), vec2(iMaxProbesPerLayer);
             for (int p = iMaxProbesPerLayer - 1; p >= 0; --p)
             {
                 // 1st rotation
                 IFPair ifPair = minQue1.top();
                 minQue1.pop();
                 vec1[p] = ifPair;

                 // 2nd rotation
                 ifPair = minQue2.top();
                 minQue2.pop();
                 vec2[p] = ifPair;
             }

             // Now begin building the query probes on ONE table
             for (const auto& ifPair1: vec1)
             {
                 int iBucketIndex1 = ifPair1.m_iIndex;
                 float fAbsHashValue1 = ifPair1.m_fValue;

                 //cout << "Hash index 1: " << iBucketIndex1 << " projection value: " << fAbsHashValue1 << endl;

                 for (const auto& ifPair2: vec2)         //p: probing step
                 {
                     int iBucketIndex2 = ifPair2.m_iIndex;
                     float fAbsHashValue2 = ifPair2.m_fValue;

                     //cout << "Hash index 2: " << iBucketIndex2 << " projection value: " << fAbsHashValue2 << endl;

                     // Start building the probe sequence
                     int iBucketIndex = iBucketIndex1 * (2 * FalconnPP::n_proj) + iBucketIndex2; // (totally we have 2D * 2D buckets)
                     float fSumHashValue = fAbsHashValue1 + fAbsHashValue2;

                     // assert(iBucketIndex < iTotalBuckets);

                     // IMPORTANT: Must use ALL iMaxProbesPerTable < iMaxProbesPerLayer^2
                     // since the minQueue will pop the min projection value first
                     // If do not use iMaxProbesPerLayer^2, we miss the bucket of query (max + max)
                     if ((int)vecMinQue[l].size() < iMaxProbesPerTable)
                         vecMinQue[l].emplace(iBucketIndex, fSumHashValue);

                     else if (fSumHashValue > vecMinQue[l].top().m_fValue)
                     {
                         vecMinQue[l].pop(); // pop max, and push min hash distance
                         vecMinQue[l].emplace(iBucketIndex, fSumHashValue);
                     }
                 }
             }

    //            assert((int)vecMinQue[l].size() == iMaxProbesPerTable);

         }

         /* Now vecMinQue is a vector of L tables
          * Each contains a priority queue of size iMaxProbesPerTable
          * with value as |<q, r_i>| + |<q, s_j>| and the key as bucket ID (i, j) = i * 2D + j
          * We need to dequeue to get the bucket ID on the right order
          * We have to integrate the tableID into the bucketID as we use vectorTables to store L tables, each with 4D^2 buckets
          * Every table has iMaxProbes positions for query probing
          * Store the list of prepared probing buckets in an 1D array of size L * maxProbe
          **/

         vector<IFPair> vecBucketProbes(FalconnPP::n_tables * iMaxProbesPerTable);
         for (int l = 0; l < FalconnPP::n_tables; ++l)
         {
             int iBaseTableIdx = l * numBucketsPerTable; // base table idx
             int iBaseIdx = l * iMaxProbesPerTable; // each table has exactly iMaxProbes buckets

             int idx = iMaxProbesPerTable - 1; // start from max as we dequeue

             while (!vecMinQue[l].empty())
             {
                 // m_iIndex = hashIndex, mfValue = absHashValue1 + absHashValue2
                 IFPair ifPair = vecMinQue[l].top();
                 vecMinQue[l].pop();

                 //cout << "Hash index: " << ifPair.m_iIndex << endl;

                 // Now: ifPair.m_iIndex is the hash Index (i, j) = 2D * i + j
                 // changing the index to have TableIdx information since we iterate probing through all tables
                 // This index is used to access the hash table vecTables as we keep 1D array for L tables, each with 4D^2 buckets
                 ifPair.m_iIndex = iBaseTableIdx + ifPair.m_iIndex;
                 // Now: ifPair.m_iIndex contains the position of the table idx & hashIndex


                 vecBucketProbes[iBaseIdx + idx] = ifPair; // ifPair.m_fValue is still |<q, r_i>| + |<q, s_j>|
                 idx--;
             }

    //            printVector(vecProbes[l]);

         }


         // MaxQueueProbes is a global probes among L tables
         // We first insert into MaxQueueProbes the query bucket ID
         // Then iteratively, pop the queue to have TableIdx and BucketID
         // Getting the data point in the bucket, adding the next bucket at TableIdx into MaxQueueProbes
         // If the table is probed many times until more than MaxProbes, then ignore this table

         priority_queue< IFPair, vector<IFPair> > maxQueProbes;
         for (int l = 0; l < FalconnPP::n_tables; ++l)
         {
             maxQueProbes.push(vecBucketProbes[l * iMaxProbesPerTable]); // position of query buckets over all tables
         }

         auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
         hashTime += (float)durTime.count();

         /** Querying **/
         startTime = chrono::high_resolution_clock::now();

         boost::dynamic_bitset<> bitsetHist(FalconnPP::n_points); // all 0's by default
         VectorXi vecProbeTracking = VectorXi::Zero(FalconnPP::n_tables);

         int iNumCand = 0;
         for (int probeCount = 0; probeCount < FalconnPP::qProbes; probeCount++)
         {
             iTotalProbes++;

             IFPair ifPair = maxQueProbes.top();
             maxQueProbes.pop();
    //            cout << "Probe " << iProbeCount << ": " << ifPair.m_iIndex << " " << ifPair.m_fValue << endl;

             uint32_t iBucketPos = FalconnPP::vecPair_BucketPos[ifPair.m_iIndex].first;
             uint16_t iBucketSize = FalconnPP::vecPair_BucketPos[ifPair.m_iIndex].second;

             // Update probe tracking
             int iTableIdx = ifPair.m_iIndex / numBucketsPerTable; // get table idx

             //cout << "Table: " << iTableIdx << " Hash index: " << ifPair.m_iIndex - iTableIdx * NUM_BUCKET << endl;
             //printVector(vecBucket);

             vecProbeTracking(iTableIdx)++;

             // insert into the queue for next probes
             // If the table is probed so many time until the limit of maxProbes, then ignore this table
             if (vecProbeTracking(iTableIdx) < iMaxProbesPerTable)
             {
                 // vecBucketProbes has range l * iMaxProbesPerTable + idx (ie top-probes)
                 IFPair ifPair = vecBucketProbes[iTableIdx * iMaxProbesPerTable + vecProbeTracking(iTableIdx)]; // get the next bucket idx of the investigated hash table
                 maxQueProbes.push(ifPair);
             }

             if (iBucketSize == 0)
             {
                 iNumEmptyBucket++;
                 continue;
             }

             iTotalCand += iBucketSize;

             // Get all points in the bucket
             for (int i = 0; i < iBucketSize; ++i)
             {
                 int iPointIdx = FalconnPP::vecTables_1D[iBucketPos + i];

                 // assert (iPointIdx < FalconnPP::n_points);
                 // assert (iPointIdx >= 0);

                 if (~bitsetHist[iPointIdx])
                 {
                     iNumCand++;
                     bitsetHist[iPointIdx] = true;
                 }

             }

             // Only need if there is some setting on candSize to limit # distance computations
             // if (iNumCand >= FalconnPP::candSize)
             //     break;
         }

         durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
         lookupTime += (float)durTime.count();

    //        iTotalProbes += iProbeCount;
         iTotalUniqueCand += iNumCand; // bitsetHist.count();

         startTime = chrono::high_resolution_clock::now();

    //		matTopK.col(q) = computeSimilarity(setCandidates, q);
    //		cout << "Nuber of bit set: " << setHistogram.count() << endl;

         // getTopK(bitsetHist, vecQuery, matTopK.col(q));

    //        cout << "Number of candidate: " << bitsetHist.count() << endl;
         if (iNumCand == 0)
             continue;

    //        cout << "Bug here...: " << bitsetHist.count() << endl;

         // This is to get top-K
         priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

         //new A class to do inner product by SIMD
    //        hnswlib::InnerProductSpace inner(PARAM_DATA_D);
    //        hnswlib::DISTFUNC<float> fstdistfunc_ = inner.get_dist_func();
    //        void *dist_func_param_;
    //        dist_func_param_ = inner.get_dist_func_param();


         int iPointIdx = bitsetHist.find_first();
         while (iPointIdx != boost::dynamic_bitset<>::npos)
         {
             // Get dot product
    //            float fInnerProduct = fstdistfunc_(vecQuery.data(), MATRIX_X.col(iPointIdx).data(), dist_func_param_);
             float fInnerProduct = vecQuery.dot(FalconnPP::matrix_X.col(iPointIdx));

             // Add into priority queue
             if (int(minQueTopK.size()) < n_neighbors)
                 minQueTopK.emplace(iPointIdx, fInnerProduct);

             else if (fInnerProduct > minQueTopK.top().m_fValue)
             {
                 minQueTopK.pop();
                 minQueTopK.emplace(iPointIdx, fInnerProduct);
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

         durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
         distTime += (float)durTime.count();
     }

     auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);


     if (verbose)
     {
         cout << "Average number of empty buckets per query: " << (float)iNumEmptyBucket / n_queries << endl;
         cout << "Average number of probes per query: " << (float)iTotalProbes / n_queries << endl;
         cout << "Average number of unique candidates per query: " << (float)iTotalUniqueCand / n_queries << endl;
         cout << "Average number of candidates per query: " << (float)iTotalCand / n_queries << endl;

         cout << "Hash and Probing Time: " << hashTime << " ms" << endl;
         cout << "Lookup Time: " << lookupTime << " ms" << endl;
         cout << "Distance Time: " << distTime << " ms" << endl;
         cout << "Querying Time: " << (float)durTime.count() << " ms" << endl;

         string sFileName = "1D_Top_" + int2str(n_neighbors) +
                            "_D_" + int2str(FalconnPP::n_proj) +
                            "_L_" + int2str(FalconnPP::n_tables) +
                            "_iProbe_" + int2str(FalconnPP::iProbes) +
                            "_bucketMinSize_" + int2str(FalconnPP::bucket_minSize) +
                            "_bucketScale_" + int2str((int)(FalconnPP::bucket_scale * 100)) +
                            "_qProbe_" + int2str(FalconnPP::qProbes) + ".txt";


         outputFile(matTopK, sFileName);
     }

    return matTopK.transpose();
}
/**********************************************************************************************************************/


/**
 * Used 2D vector<bucket> to support delete and insert new points in future
 * - Number of buckets are fixed: L * 4D^2
 * - Each bucket is a vector<int>
 *
 * We build 2 layers LSH, supporting for million-point data sets
 *
 * @ param matX: col-wise dataset of size D x N
 *
 */

void FalconnPP::build2Layers(const Ref<const MatrixXf> & matX){

    cout << "n_points: " << FalconnPP::n_points << endl;
    cout << "n_features: " << FalconnPP::n_features << endl;
    cout << "n_tables: " << FalconnPP::n_tables << endl;
    cout << "n_proj: " << FalconnPP::n_proj << endl;
    cout << "fhtDim: " << FalconnPP::fhtDim << endl;
    cout << "# iProbes: " << FalconnPP::iProbes << endl;
    cout << "bucket_minSize: " << FalconnPP::bucket_minSize << endl;
    cout << "bucket_scale: " << FalconnPP::bucket_scale << endl;

    // We need to center the data set, note that matrix_X is col-wise (D x N)
    VectorXf vecCenter = matX.rowwise().mean();
    FalconnPP::matrix_X = matX.array().colwise() - vecCenter.array(); // must add colwise()

//    cout << "Finish copying dataset into index" << endl;

//    cout << "Matrix input: first data point" << endl;
//    cout << matX.col(0).transpose() << endl;
//    cout << "Falconn input: first data point" << endl;
//    cout << FalconnPP::matrix_X.col(0).transpose() << endl;
//    cout << "In memory (col-major):" << endl;
//    for (int i = 0; i < 200; i++)
//        cout << *(FalconnPP::matrix_X.data() + i) << "  ";
//    cout << endl << endl;

    srand(time(NULL)); // should only be called once for random generator
    auto start = chrono::high_resolution_clock::now();

    // Since we have 2 layers, we call HD3Generator2() to generate 2 bitHD, each for each layer
    // We have L tables, each require n_rotate = 3 random rotations.
    // Each random rotation require one random sign vectors of length fhtDim
    FalconnPP::bitHD3Generator2(FalconnPP::fhtDim * FalconnPP::n_tables * FalconnPP::n_rotate);

//    cout << "Finish generate HD3" << endl;

    double dScaleData = 0.0;
    int iNumLimitedBucket = 0;

    int log2_D = log2(FalconnPP::n_proj);
    int log2_FWHT_D = log2(FalconnPP::fhtDim);

    // # bucket = (2D)^2 as each layer has 2D buckets
    // # bucket should be ~ n_points
    int numBucketsPerTable = 4 * FalconnPP::n_proj * FalconnPP::n_proj;

    // Init the 2D index
    FalconnPP::vecTables_2D = vector<IVector> (FalconnPP::n_tables * numBucketsPerTable);

    // 2 layers, each has CEOs coefficient
    // Closest random vector is [0, D) and far random vectors is [D, 2D)
    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(FalconnPP::n_threads);
#pragma omp parallel for
    for (int l = 0 ; l < FalconnPP::n_tables; ++l)
    {
//        cout << "Hash Table " << l << endl;
        int iBaseTableIdx = l * numBucketsPerTable;

        // vecMaxQue is a hash table, each element is a bucket as a priority queue, some bucket might be empty
        vector< priority_queue< IFPair, vector<IFPair> > > vecBucket_MaxQue(numBucketsPerTable);

        /**
        Build a hash table for N points
        **/
        for (int n = 0; n < FalconnPP::n_points; ++n)
        {
//            cout << n << endl;
            VectorXf rotatedX1 = VectorXf::Zero(FalconnPP::fhtDim);
            rotatedX1.segment(0, FalconnPP::n_features) = FalconnPP::matrix_X.col(n);

            VectorXf rotatedX2 = rotatedX1;

            for (int r = 0; r < FalconnPP::n_rotate; ++r)
            {
                // Multiply with random sign
                for (int d = 0; d < FalconnPP::fhtDim; ++d)
                {
                    rotatedX1(d) *= (2 * static_cast<float>(FalconnPP::bitHD1[l * FalconnPP::n_rotate * FalconnPP::fhtDim + r * FalconnPP::fhtDim + d]) - 1);
                    rotatedX2(d) *= (2 * static_cast<float>(FalconnPP::bitHD2[l * FalconnPP::n_rotate * FalconnPP::fhtDim + r * FalconnPP::fhtDim + d]) - 1);
                }

                fht_float(rotatedX1.data(), log2_FWHT_D);
                fht_float(rotatedX2.data(), log2_FWHT_D);
            }

            // This queue is used for finding top-k max hash values and hash index for iProbes on each layer
            priority_queue< IFPair, vector<IFPair>, greater<> > minQueLayer1; // 1st layer
            priority_queue< IFPair, vector<IFPair>, greater<> > minQueLayer2; // 2nd layer

//            cout << "Finish rotation" << endl;

            /**
            We reuse minQueLayer1 and minQueLayer2 to keep top-max abs projection for each layer
            **/
            for (int r = 0; r < FalconnPP::n_proj; ++r)
            {
                // 1st rotation
                int iSign = sgn(rotatedX1(r));
                float fAbsHashValue = iSign * rotatedX1(r);

                int iBucketIndex = r;
                if (iSign < 0)
                    iBucketIndex |= 1UL << log2_D; // set bit at position log2(D)

                if ((int)minQueLayer1.size() < FalconnPP::iProbes)
                    minQueLayer1.emplace(iBucketIndex, fAbsHashValue); // emplace is push without creating temp data

                // in case full queue
                else if (fAbsHashValue > minQueLayer1.top().m_fValue)
                {
                    minQueLayer1.pop();
                    minQueLayer1.emplace(iBucketIndex, fAbsHashValue); // No need IFPair
                }

                // 2nd rotation
                iSign = sgn(rotatedX2(r));
                fAbsHashValue = iSign * rotatedX2(r);

                iBucketIndex = r;
                if (iSign < 0)
                    iBucketIndex |= 1UL << log2_D; // set bit at position log2(D)

                if ((int)minQueLayer2.size() < FalconnPP::iProbes)
                    minQueLayer2.emplace(iBucketIndex, fAbsHashValue);

                    // in case full queue
                else if (fAbsHashValue > minQueLayer2.top().m_fValue)
                {
                    minQueLayer2.pop();
                    minQueLayer2.emplace(iBucketIndex, fAbsHashValue);
                }
            }

//            assert((int)minQueLayer1.size() == FalconnPP::iProbes);
//            assert((int)minQueLayer2.size() == FalconnPP::iProbes);

            // Convert to vector
            vector<IFPair> vec1(FalconnPP::iProbes);
            vector<IFPair> vec2(FalconnPP::iProbes);

            // As min value is popped first
            for (int p = FalconnPP::iProbes - 1; p >= 0; --p)
            {
                vec1[p] = minQueLayer1.top();
                minQueLayer1.pop();

                vec2[p] = minQueLayer2.top();
                minQueLayer2.pop();
            }

            /**
            Use minQue to find the top-iProbes over 2 layers via sum of 2 estimators
            Note that vec1 and vec2 are already sorted, and has length of iProbes
            **/
            priority_queue<IFPair, vector<IFPair>, greater<>> minQue;

            for (const auto& ifPair1: vec1)         //p: probing step
            {
                int iBucketIndex1 = ifPair1.m_iIndex;
                float fAbsHashValue1 = ifPair1.m_fValue;

                for (const auto& ifPair2: vec2)         //p: probing step
                {
                    int iBucketIndex2 = ifPair2.m_iIndex;
                    float fAbsSumHash = ifPair2.m_fValue + fAbsHashValue1; // sum of 2 estimators

                    int iBucketIndex = iBucketIndex1 * (2 * FalconnPP::n_proj) + iBucketIndex2; // (totally we have 2D * 2D buckets)

                    // new pair for inserting into priQueue
                    // assert(iBucketIndex < NUM_BUCKET);

                    // Push all points into the bucket
                    if ((int)minQue.size() < FalconnPP::iProbes)
                        minQue.emplace(iBucketIndex, fAbsSumHash);

                    else if (fAbsSumHash > minQue.top().m_fValue)
                    {
                        minQue.pop();
                        minQue.emplace(iBucketIndex, fAbsSumHash);
                    }
                }
            }

            /**
            Insert point (n, absProjectionValue) into a bucket as a priority queue
            We will have to extract top-percentage points in this queue later.
            **/

            while (!minQue.empty())
            {
                IFPair ifPair = minQue.top(); // index is bucketID, value is sumAbsHash
                minQue.pop();
                vecBucket_MaxQue[ifPair.m_iIndex].emplace(n, ifPair.m_fValue);
            }
        }

//        cout << "Debug: Now start adding points into the bucket" << endl;
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
        for (int h = 0; h < numBucketsPerTable; ++h )
        {
            // NOTE: must check empty bucket - otherwise crash
            if (vecBucket_MaxQue[h].empty())
            {
//                cout << h << endl; // for debug
                continue;
            }


            int iBucketSize = vecBucket_MaxQue[h].size();

            // We must scale to make sure that the number of points is: scale * N
            int iLimit = (int)ceil(FalconnPP::bucket_scale * iBucketSize / FalconnPP::iProbes);

            // For small scaled bucket (sparse area), we scale down to bucket_minSize
            // In practice, it is helpful when queries fall into sparse areas though we cannot guarantee # points in a table
            // For fair comparison with theoretical LSF with fixed threshold, we must always scale the bucket
            if (iLimit < FalconnPP::bucket_minSize)
            {
                // we do not want to scale to < bucket_minSize, and < bucketSize
                iLimit = min(FalconnPP::bucket_minSize, iBucketSize); // iBucket > iLimit
                iNumLimitedBucket++;
            }

            iNumPoint += iLimit;

            vector<int> vecBucket(iLimit, 0); // hack: use -1 to find the bug if happen

            // Since the queue pop the max value first
            for (int i = 0; i < iLimit; ++i )
            {
                vecBucket[i] = vecBucket_MaxQue[h].top().m_iIndex;
                vecBucket_MaxQue[h].pop();
            }

            // Need to clear queue to reduce memory
            while (!vecBucket_MaxQue[h].empty())
                vecBucket_MaxQue[h].pop();

            FalconnPP::vecTables_2D[iBaseTableIdx + h] = vecBucket;
        }

        dScaleData += (1.0 * iNumPoint / FalconnPP::n_points) / FalconnPP::n_tables;
    }

    //shink_to_fit
    FalconnPP::vecTables_2D.shrink_to_fit();

	cout << "Finish building index... " << endl;

    // TODO: Optimize the size of the index by considering an array of pointers, which point to a bucket (e.g. vector<int>)
    double dIndexSize = 1.0 * sizeof(FalconnPP::vecTables_2D) / (1 << 30); // size of pointer: 24 bytes
    dIndexSize += 1.0 * sizeof(FalconnPP::vecTables_2D[0]) * FalconnPP::vecTables_2D.size() / (1 << 30) ; // capacity() ?
    cout << "Header size of vecTables_2D: " << dIndexSize << endl;

    // RAM of each array
    for (const auto & vec : FalconnPP::vecTables_2D){
        dIndexSize += 1.0 * sizeof(int) * vec.size() / (1 << 30) ; // capacity() ?
    }
    cout << "Size of Falconn++ (2 layers) index in GB: " << dIndexSize << endl;

    dIndexSize = 1.0 * sizeof(float) * FalconnPP::matrix_X.rows() * FalconnPP::matrix_X.cols() / (1 << 30); // capacity() ?
    cout << "Size of data in GB: " << dIndexSize << endl;

    cout << "n_points per table / n: " << dScaleData << endl;
    cout << "Percentage of scaled buckets in a table: " << 1.0 * iNumLimitedBucket / (numBucketsPerTable * FalconnPP::n_tables) << endl;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end - start);
    cout << "Construct Falconn++ (2 layers) Wall Time (in seconds): " << (float)duration.count() << " seconds" << endl;

}

/**
 * Query on 2 layers LSH with nested vector index FalconnPP::vecTables_2D
 *
 * TODO: Using _mm_prefetch and SIMD for faster inner product computation
 *
 * @ matQ: col-wise matrix query of size D x Q
 */
MatrixXi FalconnPP::query2Layers(const Ref<const MatrixXf> & matQ, int n_neighbors, bool verbose)
{
    int n_queries = matQ.cols();

    if (verbose)
    {
        cout << "number of queries: " << n_queries << endl;
        cout << "# qProbes: " << FalconnPP::qProbes << endl;
        cout << "number of threads: " << FalconnPP::n_threads << endl;
    }

//    cout << matQ.col(0).transpose() << endl;
//    cout << "In memory (col-major):" << endl;
//    for (int i = 0; i < 200; i++)
//        cout << *(matQ.data() + i) << "  ";
//    cout << endl << endl;

    auto startQueryTime = chrono::high_resolution_clock::now();

    float hashTime = 0, lookupTime = 0, distTime = 0;
    uint64_t iTotalProbes = 0, iTotalUniqueCand = 0, iTotalCand = 0;

    MatrixXi matTopK = MatrixXi::Zero(n_neighbors, n_queries);

    int numBucketsPerTable = 4 * FalconnPP::n_proj * FalconnPP::n_proj;
    int iNumEmptyBucket = 0;
    int log2_D = log2(FalconnPP::n_proj);
    int log2_FWHT_D = log2(FalconnPP::fhtDim);

    // Trick: Only sort to get top-maxProbe since we do not need to use the rest of values.
    // This will reduce the cost of L D logD to L D log(maxProbe) for faster querying
    // 2.0 * should have enough number of probes per rotation to extract the top-k projection values
    int iMaxProbesPerTable = ceil(2.0 * FalconnPP::qProbes / FalconnPP::n_tables);
    int iMaxProbesPerLayer = ceil(sqrt(1.0 * iMaxProbesPerTable)); // Each layer considers maxProbes

//    cout << "Max probes per table is " << iMaxProbesPerTable << endl;
//    cout << "Max probes per rotation is " << iMaxProbesPerLayer << endl;

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(FalconnPP::n_threads);
#pragma omp parallel for reduction(+:hashTime, lookupTime, distTime, iTotalProbes, iTotalUniqueCand, iTotalCand)
    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.col(q);

        // For each table, we store the top-m largest projections of |<q, r_i>| + |<q, s_j>|
        // The index (i, j) --> i * (2D) + j as each layer has 2D buckets.
        // This index will be used as the probing sequence among L tables
        // Therefore, we must keep track these (i, j) pairs on each table
        vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vecMinQue(FalconnPP::n_tables);

        /** Rotating and prepared probes sequence **/
        for (int l = 0; l < FalconnPP::n_tables; ++l)
        {
            VectorXf rotatedQ1 = VectorXf::Zero(FalconnPP::fhtDim);
            rotatedQ1.segment(0, FalconnPP::n_features) = vecQuery;

            VectorXf rotatedQ2 = rotatedQ1;

            // Apply HD3HD2HD1
            for (int r = 0; r < FalconnPP::n_rotate; ++r)
            {
                for (int d = 0; d < FalconnPP::fhtDim; ++d)
                {
                    rotatedQ1(d) *= (2 * static_cast<float>(FalconnPP::bitHD1[l * FalconnPP::n_rotate * FalconnPP::fhtDim + r * FalconnPP::fhtDim + d]) - 1);
                    rotatedQ2(d) *= (2 * static_cast<float>(FalconnPP::bitHD2[l * FalconnPP::n_rotate * FalconnPP::fhtDim + r * FalconnPP::fhtDim + d]) - 1);
                }

                fht_float(rotatedQ1.data(), log2_FWHT_D);
                fht_float(rotatedQ2.data(), log2_FWHT_D);
            }

            // Assign hashIndex and compute distance between hashValue and the maxValue
            // Then insert into priority queue
            // Get top-k max position on each rotations
            // minQueue might be better regarding space usage, hence better for cache
            priority_queue< IFPair, vector<IFPair>, greater<> > minQue1;
            priority_queue< IFPair, vector<IFPair>, greater<> > minQue2;

            for (int r = 0; r < FalconnPP::n_proj; ++r)
            {
                // 1st rotation
                int iSign = sgn(rotatedQ1(r));
                float fHashDiff = iSign * rotatedQ1(r);

                // Get hashIndex
                int iBucketIndex = r;
                if (iSign < 0)
                    iBucketIndex |= 1UL << log2_D;

                // cout << "Hash index 1 : " << iBucketIndex << endl;

                // NOTE: avgProbesPerTable = qProbes / # Tables
                // hard code on iMaxProbes = 2 * avgProbesPerTable: we only keep top-(2 * avgProbesPerTable) smallest value

                // FalconnPP uses block sorting to save sorting time
                if ((int)minQue1.size() < iMaxProbesPerLayer)
                    minQue1.emplace(iBucketIndex, fHashDiff);

                // queue is full
                else if (fHashDiff > minQue1.top().m_fValue)
                {
                    minQue1.pop(); // pop max, and push min hash distance
                    minQue1.emplace(iBucketIndex, fHashDiff);
                }

                // 2nd rotation
                iSign = sgn(rotatedQ2(r));
                fHashDiff = iSign * rotatedQ2(r);

                // Get hashIndex
                iBucketIndex = r;
                if (iSign < 0)
                    iBucketIndex |= 1UL << log2_D;

                // cout << "Hash index 2: " << iBucketIndex << endl;

                // hard code on iMaxProbes = 2 * averageProbes: we only keep 2 * averageProbes smallest value
                // FalconnPP uses block sorting to save sorting time
                if ((int)minQue2.size() < iMaxProbesPerLayer)
                    minQue2.emplace(iBucketIndex, fHashDiff);

                // queue is full
                else if (fHashDiff > minQue2.top().m_fValue)
                {
                    minQue2.pop(); // pop max, and push min hash distance
                    minQue2.emplace(iBucketIndex, fHashDiff);
                }
            }

//            assert((int)minQue1.size() == iMaxProbesPerLayer);
//            assert((int)minQue2.size() == iMaxProbesPerLayer);

            // Convert to vector, the large projection value is in [0]
            // Hence better for creating a sequence of probing since we do not have to call pop() many times
            vector<IFPair> vec1(iMaxProbesPerLayer), vec2(iMaxProbesPerLayer);
            for (int p = iMaxProbesPerLayer - 1; p >= 0; --p)
            {
                // 1st rotation
                IFPair ifPair = minQue1.top();
                minQue1.pop();
                vec1[p] = ifPair;

                // 2nd rotation
                ifPair = minQue2.top();
                minQue2.pop();
                vec2[p] = ifPair;
            }

            // Now begin building the query probes on ONE table
            for (const auto& ifPair1: vec1)
            {
                int iBucketIndex1 = ifPair1.m_iIndex;
                float fAbsHashValue1 = ifPair1.m_fValue;

                //cout << "Hash index 1: " << iBucketIndex1 << " projection value: " << fAbsHashValue1 << endl;

                for (const auto& ifPair2: vec2)         //p: probing step
                {
                    int iBucketIndex2 = ifPair2.m_iIndex;
                    float fAbsHashValue2 = ifPair2.m_fValue;

                    //cout << "Hash index 2: " << iBucketIndex2 << " projection value: " << fAbsHashValue2 << endl;

                    // Start building the probe sequence
                    int iBucketIndex = iBucketIndex1 * (2 * FalconnPP::n_proj) + iBucketIndex2; // (totally we have 2D * 2D buckets)
                    float fSumHashValue = fAbsHashValue1 + fAbsHashValue2;

                    // assert(iBucketIndex < iTotalBuckets);

                    // IMPORTANT: Must use ALL iMaxProbesPerTable < iMaxProbesPerLayer^2
                    // since the minQueue will pop the min projection value first
                    // If do not use iMaxProbesPerLayer^2, we miss the bucket of query (max + max)

                    if ((int)vecMinQue[l].size() < iMaxProbesPerTable)
                        vecMinQue[l].emplace(iBucketIndex, fSumHashValue);

                    else if (fSumHashValue > vecMinQue[l].top().m_fValue)
                    {
                        vecMinQue[l].pop(); // pop max, and push min hash distance
                        vecMinQue[l].emplace(iBucketIndex, fSumHashValue);
                    }
                }
            }

//            assert((int)vecMinQue[l].size() == iMaxProbesPerTable);

        }

        /* Now vecMinQue is a vector of L tables
         * Each contains a priority queue of size iMaxProbesPerTable
         * with value as |<q, r_i>| + |<q, s_j>| and the key as bucket ID (i, j) = i * 2D + j
         * We need to dequeue to get the bucket ID on the right order
         * We have to integrate the tableID into the bucketID as we use vectorTables to store L tables, each with 4D^2 buckets
         * Every table has iMaxProbes positions for query probing
         * Store the list of prepared probing buckets in an 1D array of size L * maxProbe
         **/

        vector<IFPair> vecBucketProbes(FalconnPP::n_tables * iMaxProbesPerTable);
        for (int l = 0; l < FalconnPP::n_tables; ++l)
        {
            int iBaseTableIdx = l * numBucketsPerTable; // base table idx
            int iBaseIdx = l * iMaxProbesPerTable; // each table has exactly iMaxProbes buckets

            int idx = iMaxProbesPerTable - 1; // start from max as we dequeue

            while (!vecMinQue[l].empty())
            {
                // m_iIndex = hashIndex, mfValue = absHashValue1 + absHashValue2
                IFPair ifPair = vecMinQue[l].top();
                vecMinQue[l].pop();

                //cout << "Hash index: " << ifPair.m_iIndex << endl;

                // Now: ifPair.m_iIndex is the hash Index (i, j) = 2D * i + j
                // changing the index to have TableIdx information since we iterate probing through all tables
                // This index is used to access the hash table vecTables as we keep 1D array for L tables, each with 4D^2 buckets
                ifPair.m_iIndex = iBaseTableIdx + ifPair.m_iIndex;
                // Now: ifPair.m_iIndex contains the position of the table idx & hashIndex


                vecBucketProbes[iBaseIdx + idx] = ifPair; // ifPair.m_fValue is still |<q, r_i>| + |<q, s_j>|
                idx--;
            }

//            printVector(vecProbes[l]);

        }


        // MaxQueueProbes is a global probes among L tables
        // We first insert into MaxQueueProbes the query bucket ID
        // Then iteratively, pop the queue to have TableIdx and BucketID
        // Getting the data point in the bucket, adding the next bucket at TableIdx into MaxQueueProbes
        // If the table is probed many times until more than MaxProbes, then ignore this table

        priority_queue< IFPair, vector<IFPair> > maxQueProbes;
        for (int l = 0; l < FalconnPP::n_tables; ++l)
        {
            maxQueProbes.push(vecBucketProbes[l * iMaxProbesPerTable]); // position of query buckets over all tables
        }

        auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
        hashTime += (float)durTime.count();

        /** Querying **/
        startTime = chrono::high_resolution_clock::now();

        boost::dynamic_bitset<> bitsetHist(FalconnPP::n_points); // all 0's by default
        VectorXi vecProbeTracking = VectorXi::Zero(FalconnPP::n_tables);

        int iNumCand = 0;
        for (int probeCount = 0; probeCount < FalconnPP::qProbes; probeCount++)
        {
            iTotalProbes++;

            IFPair ifPair = maxQueProbes.top();
            maxQueProbes.pop();
//            cout << "Probe " << iProbeCount << ": " << ifPair.m_iIndex << " " << ifPair.m_fValue << endl;

            IVector vecBucket = FalconnPP::vecTables_2D[ifPair.m_iIndex];

            // Update probe tracking
            int iTableIdx = ifPair.m_iIndex / numBucketsPerTable; // get table idx

            //cout << "Table: " << iTableIdx << " Hash index: " << ifPair.m_iIndex - iTableIdx * numBucketsPerTable << endl;
            //printVector(vecBucket);

            vecProbeTracking(iTableIdx)++;

            // insert into the global qProbing queue for next probes on this table
            // If the table is probed so many time until the limit of maxProbes, then ignore this table
            if (vecProbeTracking(iTableIdx) < iMaxProbesPerTable)
            {
                // vecBucketProbes has range l * iMaxProbesPerTable + idx (ie top-probes)
                IFPair ifPair = vecBucketProbes[iTableIdx * iMaxProbesPerTable + vecProbeTracking(iTableIdx)]; // get the next bucket idx of the investigated hash table
                maxQueProbes.push(ifPair);
            }

            if (vecBucket.empty())
            {
                iNumEmptyBucket++;
                continue;
            }

            iTotalCand += vecBucket.size();

            // Get all points in the bucket
            for (int iPointIdx : vecBucket)
            {
//                assert (iPointIdx < FalconnPP::n_points);
//                assert (iPointIdx >= 0);

                if (~bitsetHist[iPointIdx])
                {
                    iNumCand++;
                    bitsetHist[iPointIdx] = true;
                }

            }

            // Only need if there is some setting on candSize to limit # distance computations
            // if (iNumCand >= FalconnPP::candSize)
            //     break;
        }

        durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
        lookupTime += (float)durTime.count();

//        iTotalProbes += iProbeCount;
        iTotalUniqueCand += iNumCand; // bitsetHist.count();

        startTime = chrono::high_resolution_clock::now();

//		matTopK.col(q) = computeSimilarity(setCandidates, q);
//		cout << "Number of bit set: " << setHistogram.count() << endl;

        // getTopK(bitsetHist, vecQuery, matTopK.col(q));

//        cout << "Number of candidate: " << bitsetHist.count() << endl;
        if (iNumCand == 0)
            continue;

//        cout << "Bug here...: " << bitsetHist.count() << endl;

        // This is to get top-K
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

        int iPointIdx = bitsetHist.find_first();
        while (iPointIdx != boost::dynamic_bitset<>::npos)
        {
//            if (q == 0)
//                cout << iPointIdx << " and " << vecQuery.dot(FalconnPP::matrix_X.col(iPointIdx)) << endl;

            // TODO: Faster inner product with manual prefetch and SIMD
            float fInnerProduct = vecQuery.dot(FalconnPP::matrix_X.col(iPointIdx));

            // Add into priority queue
            if (int(minQueTopK.size()) < n_neighbors)
                minQueTopK.emplace(iPointIdx, fInnerProduct);

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.emplace(iPointIdx, fInnerProduct);
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

        durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime);
        distTime += (float)durTime.count();
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);

    if (verbose)
    {
        // Print out intermediate information of querying process

        cout << "Average number of empty buckets per query: " << (float) iNumEmptyBucket / n_queries << endl;
        cout << "Average number of probes per query: " << (float) iTotalProbes / n_queries << endl;
        cout << "Average number of unique candidates per query: " << (float) iTotalUniqueCand / n_queries
             << endl;
        cout << "Average number of candidates per query: " << (float) iTotalCand / n_queries << endl;

        cout << "Hash and Probing Time: " << hashTime << " ms" << endl;
        cout << "Lookup Time: " << lookupTime << " ms" << endl;
        cout << "Distance Time: " << distTime << " ms" << endl;
        cout << "Querying Time: " << (float)durTime.count() << " ms" << endl;

        string sFileName = "2D_Top_" + int2str(n_neighbors) +
                           "_D_" + int2str(FalconnPP::n_proj) +
                           "_L_" + int2str(FalconnPP::n_tables) +
                           "_iProbe_" + int2str(FalconnPP::iProbes) +
                           "_bucketMinSize_" + int2str(FalconnPP::bucket_minSize) +
                           "_bucketScale_" + int2str((int)(FalconnPP::bucket_scale * 100)) +
                           "_qProbe_" + int2str(FalconnPP::qProbes) + ".txt";


        outputFile(matTopK, sFileName);
    }

    return matTopK.transpose();
}




