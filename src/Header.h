#pragma once

#include "fht.h"

#include <Eigen/Dense>
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

struct IndexParam
{
    int numPoints;
    int numDim;
    int bucketMinSize;
    float bucketScale;
    int numTables;
    int numProj;
    int iProbes;
    int numThreads;
};

struct QueryParam{

    int numQueries;
    int topK;
    int qProbes;
    bool verbose;
};
