#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include "Header.h"

//#include <fstream> // fscanf, fopen, ofstream
//#include <sstream> // stringstream
//#include <time.h> // for time(0) to generate different random number
//#include <cmath> // for cdf of normal distribution

//#include <immintrin.h> // Include header for SIMD intrinsics

/**
Convert an integer to string
**/
inline string int2str(int x)
{
    stringstream ss;
    ss << x;
    return ss.str();
}

/**
Get sign
**/
inline int sgn(float x)
{
    if (x >= 0) return 1;
    else return -1;
    // return 0;
}

/**
 * Save output
 */
void outputFile(const Ref<const MatrixXi> & , const string & );


void readIndexParam(int, char**, IndexParam & );
void readQueryParam(int, char** , QueryParam &);
void loadtxtData(const string &, int , int, MatrixXf & );

// Fast Hadamard transform
/**
    Convert a = a + b, b = a - b
**/
// void inline wht_bfly (float& a, float& b)
// {
//     float tmp = a;
//     a += b;
//     b = tmp - b;
// }

/**
    Fast in-place Walsh-Hadamard Transform (http://www.musicdsp.org/showone.php?id=18)
    also see (http://stackoverflow.com/questions/22733444/fast-sequency-ordered-walsh-hadamard-transform/22752430#22752430)
    - Note that the running time is exactly NlogN
**/
//void FWHT (Ref<VectorXf> data, const Ref<VectorXi> p_vecHD)
//{
//    //printVector(data);
//
//    int n = (int)data.size();
//    int nlog2 = log2(n);
//
//    int l, m;
//    for (int i = 0; i < nlog2; ++i)
//    {
//        l = 1 << (i + 1);
//        for (int j = 0; j < n; j += l)
//        {
//            m = 1 << i;
//            for (int k = 0; k < m; ++k)
//            {
//                //cout << data (j + k) << endl;
//                data (j + k) = data (j + k) * p_vecHD[j + k]; // There is a bug here
//                //cout << data (j + k) << endl;
//
//                //cout << data (j + k + m) << endl;
//                data (j + k + m) = data (j + k + m) * p_vecHD[j + k + m]; // There is a bug here
//                //cout << data (j + k + m) << endl;
//
//                wht_bfly (data (j + k), data (j + k + m));
//                //cout << data (j + k) << endl;
//                //cout << data (j + k + m) << endl;
//
//            }
//
//        }
//    }
//
//    //printVector(data);
//}



//// Function to calculate popcount using SIMD instructions
//inline uint64_t new_popcountll(uint64_t x) {
//    uint64_t c1 = 0x5555555555555555llu;
//    uint64_t c2 = 0x3333333333333333llu;
//    uint64_t c4 = 0x0F0F0F0F0F0F0F0Fllu;
//
//    x -= (x >> 1) & c1;
//    x = ((x >> 2) & c2) + (x & c2);
//    x = (x + (x >> 4)) & c4;
//    x *= 0x0101010101010101llu;
//    return x >> 56;
//};

#endif // UTILITIES_H_INCLUDED
