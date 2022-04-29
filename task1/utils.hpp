#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
using std::vector;

void free2d(float** a, vector<size_t> shape);
float* init1dArray(size_t size, float defaultValue=0.0);
float** init2dArray(size_t m, size_t n, float defaultValue=0.0);
float** matmul(float** a, float** b, vector<size_t> shapeA, vector<size_t> shapeB, bool simd=false, bool cache=false);
float** matMulElement(float** a, float** b, vector<size_t> shapeA, vector<size_t> shapeB, bool generateNewCopy=true);
float** matAdd1dArray(float** a, float* b, vector<size_t> shapeA, vector<size_t> shapeB, bool generateNewCopy=true, bool simd=false);
float** matAddMatrix(float** a, float** b, vector<size_t> shapeA, vector<size_t> shapeB, bool generateNewCopy=true);
float** matMinusScalar(float** a, float b, vector<size_t> shapeA, bool generateNewCopy=true);
float** matMulScalar(float** a, float b, vector<size_t> shapeA, bool generateNewCopy=true);
float** matMinusMat(float** a, float** b, vector<size_t> shape, bool generateNewCopy=true);
float** linearLayer(float** input, float** weight, float* bias, size_t batchSize, size_t inputSize, size_t outputSize, bool simd=false, bool cache=false);
float* arrayAddScalar(float* a, float b, size_t size, bool generateNewCopy=true);
float* arrayMulScalar(float* a, float b, size_t size, bool generateNewCopy=true);
float* arrayAddArray(float* a, float* b, size_t size, bool generateNewCopy=true);
float* sum2dLine(float** a, vector<size_t> shapeA);
float* sum2dCol(float** a, vector<size_t> shapeA);
float** softmax(float** a, vector<size_t> shapeA);
float** sigmoid(float** a, vector<size_t> shapeA);
float** sigmoidDerivative(float** a, vector<size_t> shapeA);
void print1dArray(float* a, size_t size);
void print2dArray(float** a, vector<size_t> shape);
float** transpose(float** a, vector<size_t> shape);

#endif