#ifndef FNN_HPP
#define FNN_HPP

#include <bits/stdc++.h>
using std::pair;
using std::vector;

class Network {
private:
    float** weight1;
    float** weight2;
    float* bias1;
    float* bias2;
    float** weightGrad1;
    float** weightGrad2;
    float* gradBias1;
    float* gradBias2;
    bool simd;
    bool cache;
    vector<pair<float**, vector<size_t>>> tempResult;
    size_t inputSize;
    size_t hiddenSize;
    size_t outputSize;
public:
    Network(size_t inputSize, size_t hiddenSize, size_t outputSize, bool simd = false, bool cache = false);
    ~Network();
    float** forward(float** input, size_t batchSize, bool requiredGrad=false);
    void train(float** data, float* target, size_t epoch, size_t dataSize, size_t batchSize, float learningRate);
    void clearGrad();
    void freeTempResult();
};

#endif