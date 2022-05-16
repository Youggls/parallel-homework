#ifndef FNN_HPP
#define FNN_HPP
#include "module.hpp"

class Network {
private:
    Matrix* weight1;
    Array* bias1;
    Matrix* weight2;
    Array* bias2;
    bool simd;
    size_t inputSize;
    size_t hiddenSize;
    size_t outputSize;
public:
    Network(size_t inputSize, size_t hiddenSize, size_t outputSzie, bool simd = false);
    ~Network();
    Matrix* forward(Matrix* input);
};

#endif