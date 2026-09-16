//
// Created by npha145 on 15/03/24.
//

#ifndef FALCONNPP_H
#define FALCONNPP_H

#include "header.h"

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

    int seed = -1;

    RowMatrixXf matrix_X; // n x d; points are contiguous rows

    // For 1D index, used in NeurIPS 2022
    vector<pair<uint32_t, uint16_t>> vecPair_BucketPos;
    IVector vecTables_1D;

    // For 2D index
    vector<IVector> vecTables_2D;

    int fhtDim;

    // Random signs for the two FHT layers. Layout:
    // ((table * n_rotate + rotation) * fhtDim + dimension).
    using AlignedFloatVector =
        std::vector<float, Eigen::aligned_allocator<float>>;
    AlignedFloatVector hdSigns1;
    AlignedFloatVector hdSigns2;

    inline size_t sign_offset(int table, int rotation) const noexcept
    {
        return (static_cast<size_t>(table) * n_rotate + rotation)
               * static_cast<size_t>(fhtDim);
    }

protected:

    /**
     * Generate two contiguous vectors of random signs, one for each layer.
     * The RNG draw order is layer 1 followed by layer 2 at every position.
     */
    void hdSignsGenerator2()
    {
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        if (FalconnPP::seed > -1) // then use the assigned seed
            seed = FalconnPP::seed;

        default_random_engine generator(seed);
        uniform_int_distribution<uint32_t> unifDist(0, 1);

        const size_t num_signs =
            static_cast<size_t>(n_tables) *
            static_cast<size_t>(n_rotate) *
            static_cast<size_t>(fhtDim);

        hdSigns1.resize(num_signs);
        hdSigns2.resize(num_signs);
        for (size_t i = 0; i < num_signs; ++i)
        {
            hdSigns1[i] = (unifDist(generator) & 1) ? 1.0f : -1.0f;
            hdSigns2[i] = (unifDist(generator) & 1) ? 1.0f : -1.0f;
        }
    }

public:

    // function to initialize private variables
    FalconnPP(int n, int d){
        n_points = n;
        n_features = d;
    }

    void Index2Layers(int L, int D, int bucketLimit, float alpha, int p, int t, int s) {
        n_tables = L;
        n_proj = D;
        bucket_minSize = bucketLimit;
        bucket_scale = alpha;
        iProbes = p;
        set_threads(t);
        seed = s;

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

        hdSigns1.clear();
        hdSigns2.clear();
    }

    void set_qProbes(int p){
        qProbes = p;
    }

    void set_threads(int t)
    {
        if (t <= 0)
            n_threads = omp_get_max_threads();
        else
            n_threads = t;
    }

    void build2Layers_1D(const Ref<const RowMatrixXf> &); // Used in NeurIPS 2022 for static data
    MatrixXi query2Layers_1D(const Ref<const RowMatrixXf> &, int , bool=false); // Used in NeurIPS 2022 for static data

    void build2Layers(const Ref<const RowMatrixXf> &);
    MatrixXi query2Layers(const Ref<const RowMatrixXf> &, int , bool=false);

    ~FalconnPP() { clear(); }
};

#endif //FALCONNPP_H
