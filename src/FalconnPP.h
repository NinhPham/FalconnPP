//
// Created by npha145 on 15/03/24.
//

#ifndef FALCONNPP_H
#define FALCONNPP_H

#include "Header.h"

class FalconnPP{

protected:

    int n_points;
    int n_features;

    int n_tables = 10;
    int n_proj = 512;
    int n_rotate = 3;
    int n_threads = -1;
    int bucket_minSize = 50;
    float bucket_scale = 1.0;
    int iProbes = 1;

    int qProbes = 1;

    MatrixXf matrix_X; // d x n

    // For 1D index, used in NeurIPS 2022
    vector<pair<uint32_t, uint16_t>> vecPair_BucketPos;
    IVector vecTables_1D;

    // For 2D index
    vector<IVector> vecTables_2D;

    int fhtDim;

    // 2 Layers
    boost::dynamic_bitset<> bitHD1;
    boost::dynamic_bitset<> bitHD2;

protected:

    /**
     * Generate 2 vectors of random sign, each for one layer.
     * We use boost::bitset for saving space
     * @param p_iNumBit = L * 3 * Length (3 rotation, 2 layers, each with L tables)
     */
    void bitHD3Generator2(int p_iNumBit)
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count(); // or 40 to fix random seed
        default_random_engine generator(seed);
        uniform_int_distribution<uint32_t> unifDist(0, 1);

        bitHD1 = boost::dynamic_bitset<> (p_iNumBit);
        bitHD2 = boost::dynamic_bitset<> (p_iNumBit);

        for (int d = 0; d < p_iNumBit; ++d)
        {
            bitHD1[d] = unifDist(generator) & 1;
            bitHD2[d] = unifDist(generator) & 1;
        }

//        for (int i = 0; i < 20; i++)
//        {
//            cout << bitHD1[i] << endl;
//            cout << bitHD2[i] << endl;
//        }
    }

public:

    // function to initialize private variables
    FalconnPP(int n, int d){
        n_points = n;
        n_features = d;
    }

    void Index2Layers(int L, int D, int bucketLimit, float alpha, int p, int t) {
        n_tables = L;
        n_proj = D;
        bucket_minSize = bucketLimit;
        bucket_scale = alpha;
        iProbes = p;
        n_threads = t;

        // setting fht dimension. Note n_proj must be 2^a, and > n_features
        // Ensure fhtDim > n_proj
        if (n_proj < n_features)
            fhtDim = 1 << int(ceil(log2(n_features)));
        else
            fhtDim = 1 << int(ceil(log2(n_proj)));
    }

    void clear() {
        matrix_X.resize(0, 0);
        vecTables_2D.clear();

        vecPair_BucketPos.clear();
        vecTables_1D.clear();

        bitHD1.clear();
        bitHD2.clear();
    }

    void set_qProbes(int p){
        qProbes = p;
    }

    void set_threads(int t){
        n_threads = t;
    }

    void build2Layers_1D(const Ref<const MatrixXf> &); // Used in NeurIPS 2022 for static data
    MatrixXi query2Layers_1D(const Ref<const MatrixXf> &, const int &, bool=false); // Used in NeurIPS 2022 for static data

    void build2Layers(const Ref<const MatrixXf> &);
    MatrixXi query2Layers(const Ref<const MatrixXf> &, const int &, bool=false);

    ~FalconnPP() { clear(); }
};

#endif //FALCONNPP_H
