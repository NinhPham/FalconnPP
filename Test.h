#ifndef TEST_H_INCLUDED
#define TEST_H_INCLUDED

#include "Header.h"
#include "Utilities.h"
#include "InputParser.h"

#include "BuildIndex.h"
#include "FindKNN.h"

// Fix: upD, L, scale, iProbes, candSize
// Vary: qProbe
// Scaling index, querying reads all points in the bucket
void test_FalconnCEOs2_1D_qProbes(); // using 1D scaled index
void test_thresFalconnCEOs2_1D_qProbes();

#endif
