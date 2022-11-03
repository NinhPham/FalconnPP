#include "Header.h"

#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

int PARAM_DATA_N; // Number of points (rows) of X
int PARAM_QUERY_Q; // Number of rows (queries) of Q
int PARAM_DATA_D; // Number of dimensions

int PARAM_MIPS_TOP_K; // TopK largest entries from Xq
int PARAM_MIPS_CANDIDATE_SIZE;
string PARAM_OUTPUT_FILE;

float PARAM_LSH_BUCKET_SIZE_SCALE; // Size of scale

int PARAM_LSH_NUM_TABLE;
int PARAM_LSH_NUM_PROJECTION;
int PARAM_LSH_NUM_INDEX_PROBES;
int PARAM_LSH_NUM_QUERY_PROBES;

int PARAM_INTERNAL_LSH_NUM_BUCKET;
int PARAM_INTERNAL_LOG2_NUM_PROJECTION;
int PARAM_INTERNAL_FWHT_PROJECTION; // for the case numProject < d
int PARAM_INTERNAL_LOG2_FWHT_PROJECTION;

int PARAM_TEST_LSH_L_RANGE;
int PARAM_TEST_LSH_L_BASE;
int PARAM_TEST_LSH_qPROBE_RANGE;
int PARAM_TEST_LSH_qPROBE_BASE;
float PARAM_TEST_LSH_SCALE_RANGE;
float PARAM_TEST_LSH_SCALE_BASE;


MatrixXf MATRIX_X;
MatrixXf MATRIX_Q;
boost::dynamic_bitset<> bitHD1; // all 0's by default
boost::dynamic_bitset<> bitHD2; // all 0's by default


vector<pair<uint32_t, uint16_t>> VECTOR_PAIR_FALCONN_BUCKET_POS;
vector<int> VECTOR_FALCONN_TABLES;

bool PARAM_INTERNAL_SAVE_OUTPUT = true;
bool PARAM_INTERNAL_LIMIT_BUCKET = true;
int PARAM_INTERNAL_NUM_ROTATION = 3;
