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
import falconnpp

if __name__ == '__main__':


    k = 100
    numThreads = 8

    # --------------------------------------------------------------------------------
    # Loading data set
    path = Path("~/Work/Datasets/ANNS/").expanduser()
    savePath = path / "glove200_output"

    dataset_file = path / "glove_X_norm_1183514_200.bin"
    query_file = path / "glove_Q_norm_1000_200.bin"

    nx = 1183514
    nq = 1000
    d = 200

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
    # faiss.omp_set_num_threads(numThreads)
    # t1 = timeit.default_timer()
    # bf_faiss = faiss.IndexFlatIP(d)  # build the index
    # print(bf_faiss.is_trained)
    # bf_faiss.add(X)  # add vectors to the index
    # print('Faiss BF construction time: {: .4f} in seconds'.format(timeit.default_timer() - t1))
    #
    # t1 = timeit.default_timer()
    # bf_dist, bf_ind = bf_faiss.search(Q, k)  # actual search
    # print('Faiss BF querying time: {: .4f} in seconds\n'.format(timeit.default_timer() - t1))
    # np.save(savePath / "glove200_cosine_indices_k_100.npy", bf_ind)
    # exit()

    bf_ind = np.load(savePath / "glove200_cosine_indices_k_100.npy")

    """ Falconn++ """

    # center = np.mean(dataset, axis=0) # no need centering as we will do it internally

    # index param
    numTables = 200
    numProj = 256
    bucketLimit = 50
    alpha = 0.01
    iProbes = 3
    numThreads = 8

    # Indexing
    t1 = timeit.default_timer()
    index = falconnpp.FalconnPP(nx, d)
    index.setIndexParam(numTables, numProj, bucketLimit, alpha, iProbes, numThreads)
    index.build(X)
    t2 = timeit.default_timer()
    print('Falconn++ 1D indexing time: {: .4f} in second'.format(t2 - t1))

    # might clear X for space
    
    # Querying 1D
    probeRepeats = 5
    for i in range(probeRepeats):

        t1 = timeit.default_timer()
        qProbes = 1000 * (i + 1)
        index.set_qProbes(qProbes)

        fal_ind = index.query(Q, k)
        t2 = timeit.default_timer()
        print('Falconn++ querying time: {: .4f} in seconds'.format(t2 - t1))

        fal_score = 0.0
        for q in range(nq):
            temp = len(set(bf_ind[q, :k]).intersection(set(fal_ind[q, :])))
            fal_score += float(temp) / k

        print('Falconn++ recall: {: .4f}'.format(float(fal_score) / nq))

    # --------------------------------------------------------------------------------

    # Indexing 2D
    # t1 = timeit.default_timer()
    # index.clear()
    # index.setIndexParam(numTables, numProj, bucketLimit, alpha, iProbes, numThreads)
    # index.build2D(X)
    # t2 = timeit.default_timer()
    # print('Falconn++ indexing 2D time: {}'.format(t2 - t1))

    # Querying 2D
    # index.set_threads(64)
    # for i in range(numRepeat):
    #
    #     t1 = timeit.default_timer()
    #     qProbes = 1000 * (i + 1)
    #     index.set_qProbes(qProbes)
    #
    #     fal_answers = index.query2D(Q, k)
    #     t2 = timeit.default_timer()
    #     print('Falconn++ querying time: {}'.format(t2 - t1))
    #
    #     score = 0.0
    #     for q in range(numQueries):
    #         temp = len(set(answers_bf[q, :k]).intersection(set(fal_answers[q, :])))
    #         score += float(temp) / k
    #
    #     print('Recall: {}'.format(float(score) / numQueries))

    # --------------------------------------------------------------------------------

    """ Faiss IVF """
    # nlist = 1000
    #
    # t1 = timeit.default_timer()
    # quantizer = faiss.IndexFlatL2(d)  # the other index
    #
    # #Inverted file with stored vectors.
    # # Here the inverted file pre-selects the vectors to be searched, but they are not otherwise encoded,
    # # the code array just contains the raw float entries.
    # ivf_faiss = faiss.IndexIVFFlat(quantizer, d, nlist) # by default it performs inner-product search
    #
    # assert not ivf_faiss.is_trained
    # ivf_faiss.train(X)
    # assert ivf_faiss.is_trained
    # ivf_faiss.add(X)                  # add may be a bit slower as well
    # print('IVF indexing time: {: .4f} in seconds'.format(timeit.default_timer() - t1))
    #
    # # faiss.write_index(index, path / "faiss_ivfflat_nlist_1000.bin")
    #
    # t1 = timeit.default_timer()
    # ivf_dist, ivf_ind = ivf_faiss.search(Q, k)     # actual search
    # print('Querying time w.o. probe: {: .4f} in seconds'.format(timeit.default_timer() - t1))
    #
    # ivf_score = 0.0
    # for i in range(nq):
    #     temp = len(set(bf_ind[i]).intersection(set(ivf_ind[i, :])))
    #     ivf_score += float(temp) / k
    #
    # print('IVF recall w.o. probe: {: .4f}\n'.format(float(ivf_score) / nq))
    #
    # # Test accuracy
    # probe_range = 50
    # probeRepeats = 5
    # for i in range(probeRepeats):
    #
    #     probe = probe_range * (i + 1)
    #     t1 = timeit.default_timer()
    #     ivf_faiss.nprobe = probe  # default nprobe is 1, try a few more
    #
    #     ivf_dist, ivf_ind = ivf_faiss.search(Q, k)
    #     print("Querying time with {0} probe: {1: .4f} in seconds".format(probe, timeit.default_timer() - t1))
    #
    #     ivf_score = 0.0
    #     for i in range(nq):
    #         temp = len(set(bf_ind[i]).intersection(set(ivf_ind[i, :])))
    #         ivf_score += float(temp) / k
    #
    #     print('IVF recall with {0} probe: {1: .4f}\n'.format(probe, float(ivf_score) / nq))

    """ Faiss-HNSW """
    # # Param of HNSW
    # hnsw_m = 512  # The number of neighbors for HNSW. This is typically 32
    #
    # # Setup
    # t1 = timeit.default_timer()
    # hnsw_faiss = faiss.IndexHNSWFlat(d, hnsw_m)
    # # this is the default, higher is more accurate and slower to construct
    # hnsw_faiss.hnsw.efConstruction = 200
    #
    # hnsw_faiss.verbose = True
    # hnsw_faiss.add(X)
    # t2 = timeit.default_timer()
    # print('HNSW construction time: {: .4f} in seconds'.format(t2 - t1))
    #
    # # faiss.write_index(hnsw_faiss, '/home/npha145/Dropbox/Working/_Code/_Experiment/Falconn/Glove300/Log/faiss_hnsw_ef_200_M_512.bin')
    # # index = faiss.read_index('/home/npha145/Dropbox/Working/_Code/_Experiment/Falconn/Glove300/Log/faiss_hnsw_ef_200_M_512.bin')
    #
    # ef_base = 0
    # ef_range = 25
    #
    # # Test recall
    # probeRepeat = 20
    # for i in range(probeRepeat):
    #     ef_query = ef_base + ef_range * (i + 1)
    #     # Controlling the recall for hnsw by setting ef:
    #     # higher ef leads to better accuracy, but slower search
    #
    #     print("ef_query = %d" % ef_query)
    #     hnsw_faiss.hnsw.efSearch = ef_query
    #
    #     t1 = timeit.default_timer()
    #     hnsw_dist, hnsw_ind = hnsw_faiss.search(Q, k)
    #     t2 = timeit.default_timer()
    #     print('Faiss HNSW query time: {: .4f} in seconds'.format(t2 - t1))
    #
    #     hnsw_score = 0.0
    #     for i in range(nq):
    #         temp = len(set(bf_ind[i, :k]).intersection(set(hnsw_ind[i, :])))
    #         hnsw_score += float(temp) / k
    #
    #     print('Faiss HNSW recall: {}'.format(float(hnsw_score) / nq))
