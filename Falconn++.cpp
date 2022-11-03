#include <iostream>
#include "Header.h"
#include "Utilities.h"
#include "InputParser.h"

#include "BuildIndex.h"
#include "FindKNN.h"
#include "Test.h"

#include "BF.h"

#include <time.h> // for time(0) to generate different random number
#include <stdlib.h>
#include <sys/time.h> // for gettimeofday
#include <stdio.h>
#include <unistd.h>

#include <omp.h>

int main(int nargs, char** args)
{
    srand(time(NULL)); // should only be called once for random generator

//    cout << "RAM before loading data" << endl;
//    getRAM();

    /************************************************************************/
	int iType = loadInput(nargs, args);

//    cout << "RAM after loading data" << endl;
//    getRAM();

	/************************************************************************/
	/* Approaches                                             */
	/************************************************************************/

    chrono::steady_clock::time_point begin, end;

    /************************************************************************/
	/* Algorithms                                             */
	/************************************************************************/
	switch (iType)
	{
        // Test scaled index & simple queries for 1D and 2D
        // Fix upD, L, scale, iProbes and varying qProbes
        case 91:
        {
            test_FalconnCEOs2_1D_qProbes(); // build index with 1D array
            break;

        }

        /**
        Varying scale while fixing L, iProbes, qProbes, upD
        Test only on 1D thres
        **/
        case 92:
        {
            test_thresFalconnCEOs2_1D_qProbes();
            break;
        }

        case 1: // Bruteforce topK
        {

            begin = chrono::steady_clock::now();

            BF_TopK_MIPS();

    //        getRAM();
            end = chrono::steady_clock::now();
            cout << "BF Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;

            break;
        }

        case 2: // Falconn++
        {

            // Build 1D index with fixed scale and fixed iProbes
            begin = chrono::steady_clock::now();
            scaledFalconnCEOsIndexing2_iProbes_1D(); // operating index probing
            end = chrono::steady_clock::now();
            cout << "Indexing scaled 1D_Falconn++ Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms" << endl;

    //        getRAM();
            begin = chrono::steady_clock::now();
            FalconnCEOs2_1D_TopK();
            end = chrono::steady_clock::now();
            cout << "Search scaled 1D_Falconn++ Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms" << endl;

            break;
        }

    }
}

