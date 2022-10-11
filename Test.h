#ifndef TEST_H_INCLUDED
#define TEST_H_INCLUDED

#include "Header.h"
#include "Utilities.h"
#include "InputParser.h"

#include "BuildIndex.h"
#include "FindKNN.h"

void testFalconn2_Option3(int , int );
void testFalconn2_Option2(int , int );
void testFalconn2_Option1(int , int );

void testFalconn2_L(int , int);
void testFalconn2_L_Scale(int , int , int , int );
void testFalconn2_upD(int , int);

void test2_Scale();
void test2_Scale_Pair();
void test2_Est_Scale();

void test2_1D_scaledIndex();
void test2_scaledIndex();
void test2_1D_thresIndex();

// Fix: upD, L, iProbes, candSize;
// Vary: Scale, qProbe
// Build index once, then scaling bucket size at query time
void test2_Scale_qProbes(); // use bitset to store colliding point, hence O(n^\rho)
void test2_Est_Scale_qProbes();  // using vector to store estimation, hence O(n)

// Fix: upD, L, scale, iProbes, candSize
// Vary: qProbe
// Scaling index, querying reads all points in the bucket
void test2_1D_scaledIndex_qProbes(); // using 1D scaled index
void test2_2D_scaledIndex_qProbes(); // using 2D dynamic scaled index

#endif
