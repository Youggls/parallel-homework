#include "fnn.hpp"
#include <iostream>

Network::Network(size_t inputSize, size_t hiddenSize, size_t outputSize, bool simd) {
    this->inputSize = inputSize;
    this->hiddenSize = hiddenSize;
    this->outputSize = outputSize;
    this->simd = simd;
    this->weight1 = new Matrix({inputSize, hiddenSize});
    this->weight2 = new Matrix({hiddenSize, outputSize});
    this->bias1 = new Array({hiddenSize});
    this->bias2 = new Array({outputSize});
    this->weight1->setOnes();
    this->weight2->setOnes();
    this->bias1->setZeros();
    this->bias2->setZeros();
    Config::setSimd(simd);
}

Network::~Network() {
    delete[] this->weight1;
    delete[] this->weight2;
    delete[] this->bias1;
    delete[] this->bias2;
}

Matrix* Network::forward(Matrix* input) {
    Matrix* temp1 = *((*input) * (*this->weight1)) + (*this->bias1);
    Matrix* temp2 = *((*temp1) * (*this->weight2)) + (*this->bias2);
    return temp2;
}
