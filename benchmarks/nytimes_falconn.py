from __future__ import print_function
import numpy as np
import random
import timeit
import sys
import math
import os

from pathlib import Path
import utils
import faiss
import falconn

if __name__ == '__main__':


    k = 20
    numThreads = 8
    numRepeat = 1

    # --------------------------------------------------------------------------------
    # Loading data set
    path = Path("~/Work/Datasets/ANNS/").expanduser()
    dataset_file = path / "NYTimes_X_289761_256_t.bin"
    query_file = path / "NYTimes_Q_1000_256_t.bin"

    nx = 289761
    nq = 1000
    d = 256

    X = utils.mmap_bin(dataset_file, nx, d)
    X = X.astype(np.float32)
    print("Finish reading data")

    Q = utils.mmap_bin(query_file, nq, d)
    Q = Q.astype(np.float32)
    print("Finish reading queries")

    # Dataset check
    # utils.inspect_data(X)
    # utils.inspect_data(Q)
    # exit()

    # --------------------------------------------------------------------------------
    """ Faiss BF """
    faiss.omp_set_num_threads(numThreads)
    t1 = timeit.default_timer()
    bf_faiss = faiss.IndexFlatIP(d)  # build the index
    print(bf_faiss.is_trained)
    bf_faiss.add(X)  # add vectors to the index
    print('Faiss BF construction time: {: .4f} in seconds'.format(timeit.default_timer() - t1))

    t1 = timeit.default_timer()
    bf_dist, bf_ind = bf_faiss.search(Q, k)  # actual search
    print('Faiss BF querying time: {: .4f} in seconds\n'.format(timeit.default_timer() - t1))

    """ Faiss IVF """
    nlist = 1000

    t1 = timeit.default_timer()
    quantizer = faiss.IndexFlatL2(d)  # the other index

    #Inverted file with stored vectors.
    # Here the inverted file pre-selects the vectors to be searched, but they are not otherwise encoded,
    # the code array just contains the raw float entries.
    ivf_faiss = faiss.IndexIVFFlat(quantizer, d, nlist) # by default it performs inner-product search

    assert not ivf_faiss.is_trained
    ivf_faiss.train(X)
    assert ivf_faiss.is_trained
    ivf_faiss.add(X)                  # add may be a bit slower as well
    print('IVF indexing time: {: .4f} in seconds'.format(timeit.default_timer() - t1))

    # faiss.write_index(index, path / "faiss_ivfflat_nlist_1000.bin")

    t1 = timeit.default_timer()
    ivf_dist, ivf_ind = ivf_faiss.search(Q, k)     # actual search
    print('Querying time w.o. probe: {: .4f} in seconds'.format(timeit.default_timer() - t1))

    ivf_score = 0.0
    for i in range(nq):
        temp = len(set(bf_ind[i]).intersection(set(ivf_ind[i, :])))
        ivf_score += float(temp) / k

    print('IVF recall w.o. probe: {: .4f}\n'.format(float(ivf_score) / nq))

    # Test accuracy
    probe_range = 50
    probeRepeats = 5
    for i in range(probeRepeats):

        probe = probe_range * (i + 1)
        t1 = timeit.default_timer()
        ivf_faiss.nprobe = probe  # default nprobe is 1, try a few more

        ivf_dist, ivf_ind = ivf_faiss.search(Q, k)
        print("Querying time with {0} probe: {1: .4f} in seconds".format(probe, timeit.default_timer() - t1))

        ivf_score = 0.0
        for i in range(nq):
            temp = len(set(bf_ind[i]).intersection(set(ivf_ind[i, :])))
            ivf_score += float(temp) / k

        print('IVF recall with {0} probe: {1: .4f}\n'.format(probe, float(ivf_score) / nq))

    """ Falconn (does not support multi-threads) """

    number_of_tables = 200
    number_of_bits = 18

    # Setting Falconn parameters
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = d
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

    print('Constructing Falconn LSH tables')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(X)
    print('Falconn indexing time: {: .4f} in seconds'.format(timeit.default_timer() - t1))

    query_object = table.construct_query_object()

    probeRepeats = 5
    for j in range(probeRepeats):
        number_of_probes = 1000 * (j + 1)

        # final evaluation
        query_object.set_num_probes(number_of_probes)
        query_object.reset_query_statistics()

        t1 = timeit.default_timer()
        answerLSH = []
        for Qi in Q:
            answerLSH.append(query_object.find_k_nearest_neighbors(Qi, k))
        t2 = timeit.default_timer()

        score = 0
        for i in range(nq):
            exact = bf_ind[i, :k]
            lsh = answerLSH[i]
            temp = len(set(exact).intersection(set(lsh)))
            score += float(temp) / k

        print("L = %d, number of bits = %d, number of qProbes = %d" % (
            number_of_tables, number_of_bits, number_of_probes))
        print('Query time: {: .4f} in seconds'.format((t2 - t1)))
        print('Avg precision: {: .4f}'.format(float(score) / len(Q)))

        print('Query statistics:')
        stats = query_object.get_query_statistics()
        print('Average total query time: {:e} seconds'.format(
            stats.average_total_query_time))
        print('Average LSH time: {:e} seconds'.format(
            stats.average_lsh_time))
        print('Average lookup time:  {:e} seconds'.format(
            stats.average_hash_table_time))
        print('Average distance time:    {:e} seconds'.format(
            stats.average_distance_time))
        print('Average number of candidates:        {}'.format(
            stats.average_num_candidates))
        print('Average number of unique candidates: {}\n'.format(
            stats.average_num_unique_candidates))

