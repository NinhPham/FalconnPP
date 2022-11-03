#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include "Header.h"

#include <sstream> // stringstream
#include <time.h> // for time(0) to generate different random number
#include <cmath> // for cdf of normal distribution

#include "sys/types.h" // for getting RAM
//#include "sys/sysinfo.h" // for getting RAM
//
//// Inline functions
//inline void getRAM()
//{
//    // https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
//    struct sysinfo memInfo;
//    sysinfo (&memInfo);
//    long long totalPhysMem = memInfo.totalram - memInfo.freeram;
//    //Multiply in next statement to avoid int overflow on right hand side...
//    totalPhysMem *= memInfo.mem_unit;
//    cout << "Amount of RAM: " << (float)totalPhysMem / (1L << 30) << " GB." << endl;
//}

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

// https://www.johndcook.com/blog/cpp_phi_inverse/
inline float RationalApproximation(float t)
{
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    float c[] = {2.515517, 0.802853, 0.010328};
    float d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) /
               (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

inline float NormalCDFInverse(float p)
{
    if (p <= 0.0 || p >= 1.0)
    {
        std::stringstream os;
        os << "Invalid input argument (" << p
           << "); must be larger than 0 but less than 1.";
        throw std::invalid_argument( os.str() );
    }

    // See article above for explanation of this section.
    if (p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -RationalApproximation( sqrt(-2.0*log(p)) );
    }
    else
    {
        // F^-1(p) = G^-1(1-p)
        return RationalApproximation( sqrt(-2.0*log(1-p)) );
    }
}

/* Generate Hadamard matrix
*/
void bitHD3Generator2(int);

// Saving
void outputFile(const Ref<const MatrixXi> &, string);

void clearFalconnIndex();

#endif // UTILITIES_H_INCLUDED
