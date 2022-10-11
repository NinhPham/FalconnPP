#include "Header.h"

#include <set>
#include <unordered_map>
#include <Eigen/Dense>
#include <vector>
using namespace Eigen;
using namespace std;

int PARAM_DATA_N; // Number of points (rows) of X
int PARAM_QUERY_Q; // Number of rows (queries) of Q
int PARAM_DATA_D; // Number of dimensions

int PARAM_MIPS_TOP_K; // TopK largest entries from Xq
int PARAM_MIPS_CANDIDATE_SIZE;

int PARAM_LSH_BUCKET_SIZE_LIMIT; // Size of bucket
float PARAM_LSH_BUCKET_SIZE_SCALE; // Size of scale
float PARAM_LSH_DISCARD_T; // Threshold to discard
int PARAM_LSH_NUM_TABLE;
int PARAM_LSH_NUM_PROJECTION;
int PARAM_LSH_NUM_INDEX_PROBES;
int PARAM_LSH_NUM_QUERY_PROBES;
bool PARAM_LSH_PROBING_HEURISTIC = 1;

bool PARAM_INTERNAL_LSH_FIXING_BUCKET_SIZE;
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


int PARAM_INTERNAL_LSH_RANGE;
int PARAM_INTERNAL_LSH_BASE;

MatrixXf MATRIX_X;
MatrixXf MATRIX_Q;
boost::dynamic_bitset<> bitHD1; // all 0's by default
boost::dynamic_bitset<> bitHD2; // all 0's by default

//MatrixXi MATRIX_HADAMARD;
MatrixXf MATRIX_HD1;
MatrixXf MATRIX_HD2;

//boost::multi_array<int, 3> VECTOR3D_FALCONN_TABLES;
IVector VECTOR3D_FALCONN_TABLES;
vector<IVector> VECTOR2D_FALCONN_TABLES;
vector<vector<IFPair>> VECTOR2D_PAIR_FALCONN_TABLES;

vector<pair<uint32_t, uint16_t>> VECTOR_PAIR_FALCONN_BUCKET_POS;

vector<int> VECTOR_FALCONN_TABLES;
vector<IFPair> VECTOR_PAIR_FALCONN_TABLES;



//bool INDEX_PROBING = false; //index probing flag
//bool QUERY_PROBING = false;



bool PARAM_INTERNAL_SAVE_OUTPUT = true;
bool PARAM_INTERNAL_LIMIT_BUCKET = true;
int PARAM_NUM_ROTATION = 1;
