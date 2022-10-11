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

    PARAM_INTERNAL_SAVE_OUTPUT = true; // saving results
    PARAM_NUM_ROTATION = 3;
    PARAM_INTERNAL_LIMIT_BUCKET = true;

	/************************************************************************/
	/* Approaches                                             */
	/************************************************************************/
    struct timeval startTime;
    gettimeofday(&startTime, NULL);

    //double dStart;
    chrono::steady_clock::time_point begin, end;


    /************************************************************************/
	/* Algorithms                                             */
	/************************************************************************/
	switch (iType)
	{
        // Test scaled index & simple queries for 1D and 2D
        // Fix upD, L, scale, iProbes and varying qProbes
        case 993:
        {
            test2_1D_scaledIndex_qProbes(); // build index with 1D array
//            test2_2D_scaledIndex_qProbes(); // build index with 2D array: faster but need larger space and query time
            break;

        }

        /**
        Varying scale while fixing L, iProbes, qProbes, upD
        Test only on 1D thres
        **/
        case 997:
        {
            test2_1D_thresIndex();
            break;
        }

        /**
        Bruteforce
        **/
        case 11: // Bruteforce topK
        {

            begin = chrono::steady_clock::now();

            BF_TopK_MIPS();

    //        getRAM();
            end = chrono::steady_clock::now();
            cout << "BF Wall Clock = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[us]" << endl;
            //printf("BF CPU time is %f \n", getCPUTime(clock() - dStart));

//            gettimeofday(&endTime, NULL);
//
//            seconds  = endTime.tv_sec  - startTime.tv_sec;  // second
//            useconds = endTime.tv_usec - startTime.tv_usec; // us = 0.001 ms
//
//            mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5; // in milisecond
//
//            printf("Ubuntu BF Wall time: %ld milliseconds\n", mtime);

            break;
        }
    }
}

