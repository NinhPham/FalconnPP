from __future__ import print_function
import numpy as np
import random
import timeit
import sys
import math
import os

# Falconn
def callFalconn(dataset, queries, k):

    numRepeat = 5
    numThreads = 64

    # Falconn param
    number_of_tables = 50
    number_of_bits = 18
    centering = 0

    if centering:
        center = np.mean(dataset, axis=0)
        print('We center data')
        dataset_fal = dataset - center
        print('We center query')
        queries_fal = queries - center
    else:
        dataset_fal = dataset
        queries_fal = queries

    import falconn
    # Setting Falconn parameters
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = numDim
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
    params_cp.l = number_of_tables
    # params_cp.k = 1
    # params_cp.last_cp_dimension = 512
    # we set one rotation, since the data is dense enough,
    # for sparse data set it to 2
    params_cp.num_rotations = 3
    params_cp.seed = random.randrange(sys.maxsize)  # 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = numThreads
    params_cp.storage_hash_table = falconn.StorageHashTable.FlatHashTable # falconn.StorageHashTable.BitPackedFlatHashTable
    # we build 18-bit hashes so that each table has
    # 2^18 bins; this is a good choise since 2^18 is of the same
    # order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(number_of_bits, params_cp)

    print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset_fal)
    t2 = timeit.default_timer()
    print('Construction time: {}'.format(t2 - t1))

    query_object = table.construct_query_object()

    for j in range(numRepeat):
        number_of_probes = 1000 * (j + 1)

        # final evaluation
        query_object.set_num_probes(number_of_probes)
        query_object.reset_query_statistics()

        t1 = timeit.default_timer()
        answerLSH = []
        for query in queries:
            answerLSH.append(query_object.find_k_nearest_neighbors(query, k))
        t2 = timeit.default_timer()

        score = 0
        for i in range(numQueries):
            exact = answers_bf[i, :k]
            lsh = answerLSH[i]
            temp = len(set(exact).intersection(set(lsh)))
            score += float(temp) / k

        print("L = %d, number of bits = %d, number of qProbes = %d" % (
        number_of_tables, number_of_bits, number_of_probes))
        print('Query time: {}'.format((t2 - t1)))
        print('Avg precision: {}'.format(float(score) / len(queries)))

        print('Query statistics:')
        stats = query_object.get_query_statistics()
        print('Average total query time: {:e} seconds'.format(
            stats.average_total_query_time))
        print('Average LSH time:         {:e} seconds'.format(
            stats.average_lsh_time))
        print('Average lookup time:  {:e} seconds'.format(
            stats.average_hash_table_time))
        print('Average distance time:    {:e} seconds'.format(
            stats.average_distance_time))
        print('Average number of candidates:        {}'.format(
            stats.average_num_candidates))
        print('Average number of unique candidates: {}\n'.format(
            stats.average_num_unique_candidates))

if __name__ == '__main__':


    k = 20
    numThreads = 64
    numRepeat = 5

    # --------------------------------------------------------------------------------
    # Loading data set
    dataset_file = '/home/npha145/Dataset/ANNS/CosineKNN/Glove_X_1183514_200.npy'
    query_file = '/home/npha145/Dataset/ANNS/CosineKNN/Glove_Q_1000_200.npy'

    dataset = np.load(dataset_file)
    queries = np.load(query_file)

    # --------------------------------------------------------------------------------
    numQueries = len(queries)
    numPoints, numDim = np.shape(dataset)
    # print('numPoints =', numPoints, ', numDim =', numDim, ', numQueries =', numQueries)

    # --------------------------------------------------------------------------------
    # BF
    import faiss
    faiss.omp_set_num_threads(numThreads)
    t1 = timeit.default_timer()
    index = faiss.IndexFlatL2(numDim)  # build the index
    print(index.is_trained)
    index.add(dataset)  # add vectors to the index
    t2 = timeit.default_timer()
    print('Faiss BF Construction time: {}'.format(t2 - t1))

    t1 = timeit.default_timer()
    D, answers_bf = index.search(queries, k)  # actual search
    t2 = timeit.default_timer()
    print('Faiss BF Querying time: {}'.format(t2 - t1))

    # --------------------------------------------------------------------------------
    # Falconn
    # callFalconn(dataset, queries, k)

    # --------------------------------------------------------------------------------

    # Important: Transpose dataset and queries as Falconn++ takes input as D x N, and D x Q
    # center = np.mean(dataset, axis=0)
    dataset_t = np.transpose(dataset) # no need centering as we will do it internally
    queries_t = np.transpose(queries)
    assert dataset_t.dtype == np.float32
    assert queries_t.dtype == np.float32

    # index param
    numTables = 350
    numProj = 256
    bucketLimit = 50
    alpha = 0.01
    iProbes = 3

    # Indexing
    t1 = timeit.default_timer()
    import FalconnPP

    index = FalconnPP.FalconnPP(numPoints, numDim)
    index.setIndexParam(numTables, numProj, bucketLimit, alpha, iProbes, numThreads)
    index.build(dataset_t)  # add vectors to the index, must transpose to D x N
    t2 = timeit.default_timer()
    print('Falconn++ 1D indexing time: {}'.format(t2 - t1))

    # Querying 1D
    index.set_threads(64)
    for i in range(numRepeat):

        t1 = timeit.default_timer()
        qProbes = 1000 * (i + 1)
        index.set_qProbes(qProbes)

        fal_answers = index.query(queries_t, k)
        t2 = timeit.default_timer()
        print('Falconn++ querying time: {}'.format(t2 - t1))

        score = 0.0
        for q in range(numQueries):
            temp = len(set(answers_bf[q, :k]).intersection(set(fal_answers[q, :])))
            score += float(temp) / k

        print('Recall: {}'.format(float(score) / numQueries))

    # --------------------------------------------------------------------------------

    # Indexing 2D
    t1 = timeit.default_timer()
    index.clear()
    index.setIndexParam(numTables, numProj, bucketLimit, alpha, iProbes, numThreads)
    index.build2D(dataset_t)  # add vectors to the index, must transpose to D x N
    t2 = timeit.default_timer()
    print('Falconn++ indexing 2D time: {}'.format(t2 - t1))

    # Querying 2D
    # index.set_threads(64)
    for i in range(numRepeat):

        t1 = timeit.default_timer()
        qProbes = 1000 * (i + 1)
        index.set_qProbes(qProbes)

        fal_answers = index.query2D(queries_t, k)
        t2 = timeit.default_timer()
        print('Falconn++ querying time: {}'.format(t2 - t1))

        score = 0.0
        for q in range(numQueries):
            temp = len(set(answers_bf[q, :k]).intersection(set(fal_answers[q, :])))
            score += float(temp) / k

        print('Recall: {}'.format(float(score) / numQueries))

    # --------------------------------------------------------------------------------
