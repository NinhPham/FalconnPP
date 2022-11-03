#pragma once

#include "Header.h"
#include "Utilities.h"

// Falconn++ with 2 combined LSH, using 1D array to store index
void scaledFalconnCEOsIndexing2_iProbes_1D();

// Theoretical LSF with 2 combined LSH, using 1D array to store index
// Select global threshold to filter to ensure each table has (\alpha * n) points
void thresFalconnCEOsIndexing2_1D();





