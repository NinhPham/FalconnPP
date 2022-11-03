
#include "Test.h"

/**
Build 1D scaled index then run query with different qProbes -- measure speed
Fix upD, L, scale, iProbes
Vary qProbes
**/
void test_FalconnCEOs2_1D_qProbes()
{
    chrono::steady_clock::time_point begin, end;

    clearFalconnIndex();

    //cout << "RAM before index" << endl;
    //getRAM();

    // Build 1D index with fixed scale and fixed iProbes
    begin = chrono::steady_clock::now();
    scaledFalconnCEOsIndexing2_iProbes_1D(); // operating index probing
    end = chrono::steady_clock::now();
    cout << "Indexing scaled 1D_Falconn++ Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms" << endl;

    //cout << "RAM after index" << endl;
    //getRAM();

    for (int i = 1; i <= 20; ++i)
    {
        // Varying qProbes
        PARAM_LSH_NUM_QUERY_PROBES = PARAM_TEST_LSH_qPROBE_BASE + PARAM_TEST_LSH_qPROBE_RANGE * i;
        cout << "qProbes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

        begin = chrono::steady_clock::now();
        FalconnCEOs2_1D_TopK();
        end = chrono::steady_clock::now();
        cout << "Search scaled 1D_Falconn++ Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms" << endl;
    }
}


/**
Build index then run query at the same time with different scales -- measure speed
Fix upD, L, qProbes, iProbes
Vary scale
**/
void test_thresFalconnCEOs2_1D_qProbes()
{
    chrono::steady_clock::time_point begin, end;
    vector<float> alpha_list = {0.1, 0.5};
    for (float alpha : alpha_list)
    {
        clearFalconnIndex();

        PARAM_LSH_BUCKET_SIZE_SCALE = alpha;
        cout << "Scale: " << PARAM_LSH_BUCKET_SIZE_SCALE << endl;

        //dStart = clock();
        begin = chrono::steady_clock::now();
        thresFalconnCEOsIndexing2_1D(); // operating index probing
        end = chrono::steady_clock::now();
        cout << "Indexing thres_Falconn++ Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << endl;

        for (int i = 1; i <= 20; ++i)
        {
            // Varying qProbes
            PARAM_LSH_NUM_QUERY_PROBES = PARAM_TEST_LSH_qPROBE_BASE + PARAM_TEST_LSH_qPROBE_RANGE * i;
            cout << "qProbes: " << PARAM_LSH_NUM_QUERY_PROBES << endl;

            begin = chrono::steady_clock::now();
            FalconnCEOs2_1D_TopK();
            end = chrono::steady_clock::now();
            cout << "Search scaled 1D_Falconn++ Wall Clock = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms" << endl;
        }
    }
}


