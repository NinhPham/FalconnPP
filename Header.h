#pragma once

#include "fht.h"

#include <Eigen/Dense>
#include <set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <random>

#include <chrono>
#include <iostream> // cin, cout

//#include <boost/multi_array.hpp>
#include <boost/dynamic_bitset.hpp>

using namespace Eigen;
using namespace std;

typedef vector<float> FVector;
typedef vector<int> IVector;
//typedef vector<uint32_t> I32Vector;
//typedef vector<uint64_t> I64Vector;


//typedef boost::multi_array<int, 3> IVector3D;

struct myComp {

    constexpr bool operator()(
        pair<double, int> const& a,
        pair<double, int> const& b)
        const noexcept
    {
        return a.first > b.first;
    }
};
struct IFPair
{
    int m_iIndex;
    float	m_fValue;

    IFPair()
    {
        m_iIndex = 0;
        m_fValue = 0.0;
    }

    IFPair(int p_iIndex, double p_fValue)
    {
        m_iIndex = p_iIndex;
        m_fValue = p_fValue;
    }

    // Overwrite operation < to get top K largest entries
    bool operator<(const IFPair& p) const
    {
        return m_fValue < p.m_fValue;
    }

    bool operator>(const IFPair& p) const
    {
        return m_fValue > p.m_fValue;
    }
};

extern int PARAM_DATA_N; // Number of points (rows) of X
extern int PARAM_QUERY_Q; // Number of rows (queries) of Q
extern int PARAM_DATA_D; // Number of dimensions

extern int PARAM_MIPS_TOP_K; // TopK largest entries from Xq
extern int PARAM_MIPS_CANDIDATE_SIZE;

extern int PARAM_LSH_BUCKET_SIZE_LIMIT; // Size of bucket
extern float PARAM_LSH_BUCKET_SIZE_SCALE; // Size of scale
extern float PARAM_LSH_DISCARD_T;

extern int PARAM_LSH_NUM_TABLE; // Number of hash tables
extern int PARAM_LSH_NUM_PROJECTION;
extern int PARAM_LSH_NUM_INDEX_PROBES;
extern int PARAM_LSH_NUM_QUERY_PROBES;
extern bool PARAM_LSH_PROBING_HEURISTIC;


extern bool PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE;
extern int PARAM_INTERNAL_LSH_NUM_BUCKET; // = PARAM_LSH_NUM_PROJECTION * 2
extern int PARAM_INTERNAL_LOG2_NUM_PROJECTION; // = PARAM_LSH_NUM_PROJECTION * 2
extern int PARAM_INTERNAL_FWHT_PROJECTION; // for the case numProject < d
extern int PARAM_INTERNAL_LOG2_FWHT_PROJECTION;

extern int PARAM_INTERNAL_LSH_RANGE;
extern int PARAM_INTERNAL_LSH_BASE;

extern int PARAM_TEST_LSH_L_RANGE;
extern int PARAM_TEST_LSH_L_BASE;
extern int PARAM_TEST_LSH_qPROBE_RANGE;
extern int PARAM_TEST_LSH_qPROBE_BASE;
extern float PARAM_TEST_LSH_SCALE_RANGE;
extern float PARAM_TEST_LSH_SCALE_BASE;

extern MatrixXf MATRIX_X;
extern MatrixXf MATRIX_Q;

//extern MatrixXi MATRIX_HADAMARD;
extern MatrixXf MATRIX_HD1;
extern MatrixXf MATRIX_HD2;
extern boost::dynamic_bitset<> bitHD1; // all 0's by default
extern boost::dynamic_bitset<> bitHD2; // all 0's by default

//extern boost::multi_array<int, 3> VECTOR3D_FALCONN_TABLES;
extern IVector VECTOR3D_FALCONN_TABLES;
extern vector<IVector> VECTOR2D_FALCONN_TABLES;
extern vector<vector<IFPair>> VECTOR2D_PAIR_FALCONN_TABLES;

extern vector<pair<uint32_t, uint16_t>> VECTOR_PAIR_FALCONN_BUCKET_POS;
extern vector<int> VECTOR_FALCONN_TABLES;
extern vector<IFPair> VECTOR_PAIR_FALCONN_TABLES;
// extern string PARAM_OUTPUT_FILE;

extern bool PARAM_INTERNAL_SAVE_OUTPUT;
extern bool PARAM_INTERNAL_LIMIT_BUCKET;
extern int PARAM_NUM_ROTATION;
