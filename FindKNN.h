#ifndef FINDKNN_H_INCLUDED
#define FINDKNN_H_INCLUDED

#include "Header.h"

void falconnTopK_SeqProbes();
void falconnTopK_CycProbes();

void falconnCEOsTopK_SeqProbes();
// Implement both bucketLimit and bucketScale
void falconnCEOsTopK_CycProbes();

// Simple querying: access all points in the pre-scaled bucket
// One concanated LSH for small data sets for thousands of points
void simpleFalconnCEOsTopK_CycProbes();

// Simple querying: access all points in the pre-scaled bucket
// Two conconated LSH for medium data sets for millions of points
void simpleFalconnCEOsTopK_CycProbes2();

void simpleFalconnCEOsTopK_CycProbes2_1D(); // for 1D vector
void simpleFalconnCEOsTopK_FreqProbes2_1D();
void simpleFalconnCEOsTopK_CycProbes2_1D_improve();
void simpleFalconnCEOsTopK_SeqProbes2_1D();

// Scaled querying: scale the bucket since the bucket has not been scaled yet
void scaledFalconnCEOsTopK_CycProbes2();
void scaledFalconnCEOsTopK_CycProbes2_Pair();
void scaledFalconnCEOsTopK_CycProbes2_Pair_Est();

// Thres = simple: replace the name
void thresFalconnCEOsTopK_CycProbes2_1D(); // for 1D vector

#endif
