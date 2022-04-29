#include "./fnn.hpp"
#include "./utils.hpp"

Network::Network(size_t inputSize, size_t hiddenSize, size_t outputSize, bool simd, bool cache) {
    // Set class attribute
    this->inputSize = inputSize;
    this->hiddenSize = hiddenSize;
    this->outputSize = outputSize;
    // Initialize weight and bias, all value default to 0
    this->weight1 = init2dArray(inputSize, hiddenSize);
    this->weight2 = init2dArray(hiddenSize, outputSize);
    this->bias1 = init1dArray(hiddenSize);
    this->bias2 = init1dArray(outputSize);
    this->simd = simd;
    this->cache = cache;
}

Network::~Network() {
    for (size_t i = 0; i < this->inputSize; i++) {
        delete[] this->weight1[i];
    }
    for (size_t i = 0; i < this->hiddenSize; i++) {
        delete[] this->weight2[i];
    }
    delete[] this->weight1;
    delete[] this->weight2;
    delete[] this->bias1;
    delete[] this->bias2;
}

float** Network::forward(float** input, size_t batchSize, bool requiredGrad) {
    float** hidden1 = linearLayer(input, this->weight1, this->bias1, batchSize, this->inputSize, this->hiddenSize, this->simd, this->cache);
//    float** activatedHidden1 = sigmoid(hidden1, {batchSize, this->hiddenSize});
    float** hidden2 = linearLayer(hidden1, this->weight2, this->bias2, batchSize, this->hiddenSize, this->outputSize, this->simd, this->cache);
//    float** prob = softmax(hidden2, {batchSize, this->outputSize});


    if (requiredGrad) {
        this->tempResult.push_back({hidden2, {batchSize, this->outputSize}});
  //      this->tempResult.push_back({activatedHidden1, {batchSize, this->hiddenSize}});
        this->tempResult.push_back({hidden1, {batchSize, this->hiddenSize}});
    } else {
        free2d(hidden1, {batchSize, this->hiddenSize});
    //    free2d(activatedHidden1, {batchSize, this->hiddenSize});
    }
    return hidden2;
}


void Network::clearGrad() {
    free2d(this->weightGrad1, {this->inputSize, this->hiddenSize});
    free2d(this->weightGrad2, {this->hiddenSize, this->outputSize});
    delete[] this->gradBias1;
    delete[] this->gradBias2;
    this->gradBias1 = nullptr;
    this->gradBias2 = nullptr;
}


void Network::freeTempResult() {
    for (size_t i = 0; i < this->tempResult.size(); i++) {
        free2d(this->tempResult[i].first, this->tempResult[i].second);
    }
    this->tempResult.clear();
}


void Network::train(float** data, float* target, size_t epoch, size_t dataSize, size_t batchSize, float learningRate) {
    size_t batchNumPerEpoch = dataSize / batchSize;
    if (dataSize % batchSize != 0) batchNumPerEpoch += 1;
    std::cout << "Begin to train model" << std::endl;
    std::cout << "There are total " << dataSize << " training data items" << std::endl;
    std::cout << "Learning rate: " << learningRate << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    std::cout << "Total epoch num: " << epoch << std::endl;
    std::cout << "Batch num per epoch: " << batchNumPerEpoch << std::endl;
    float** onehotTraget = init2dArray(dataSize, this->outputSize);
    for (size_t i = 0; i < dataSize; i++) {
        onehotTraget[i][int(target[i])] = 1;
    }
    for (size_t epochIndex = 0; epochIndex < epoch; epochIndex += 1) {
        std::cout << "Epoch " << epochIndex + 1 << " / " << epoch << "." << std::endl;
        for (size_t batchIndex = 0; batchIndex < batchNumPerEpoch; batchIndex += 1) {
            size_t batchStartIndex = batchIndex * batchSize;
            size_t batchEndIndex = std::min(batchStartIndex + batchSize, dataSize);
            size_t currBatchSize = batchEndIndex - batchStartIndex;
            std::cout << "Batch " << batchIndex + 1 << " / " << batchNumPerEpoch << "." << std::endl;

            float** proba = this->forward(data + batchStartIndex, currBatchSize, true);

            float** grad1 = matMinusMat(proba, onehotTraget + batchStartIndex, {currBatchSize, this->outputSize});
            // Linear2 gradient calculation
            this->gradBias2 = sum2dCol(grad1, {currBatchSize, this->outputSize});
            float** activatedHidden1Transposed = transpose(this->tempResult[1].first, this->tempResult[1].second);
            this->weightGrad2 = matmul(activatedHidden1Transposed, grad1, {this->hiddenSize, currBatchSize}, {currBatchSize, this->outputSize});
            // Linear1 gradient calculation
            float** weight2Transposed = transpose(this->weight2, {this->hiddenSize, this->outputSize});
            float** grad2Linear1 = matmul(grad1, weight2Transposed, {currBatchSize, this->outputSize}, {this->outputSize, this->hiddenSize});
            grad2Linear1 = matMulElement(grad2Linear1, this->tempResult[1].first, {currBatchSize, this->hiddenSize}, {currBatchSize, this->hiddenSize}, false);
            float** grad2Activated1 = matMulElement(
                grad2Linear1,
                sigmoidDerivative(this->tempResult[2].first, {currBatchSize, this->hiddenSize}),
                {currBatchSize, this->hiddenSize},
                {currBatchSize, this->hiddenSize},
                false
            );
            float** inputTransposed = transpose(data + batchStartIndex, {currBatchSize, this->inputSize});
            this->weightGrad1 = matmul(inputTransposed, grad2Activated1, {this->inputSize, currBatchSize}, {currBatchSize, this->hiddenSize});
            this->gradBias1 = sum2dCol(grad2Activated1, {currBatchSize, this->hiddenSize});
            // SGD update
            float* bias2Updated = arrayMulScalar(this->gradBias2, -learningRate, this->outputSize, true);
            float** weight2Updated = matMulScalar(this->weightGrad2, -learningRate, {this->hiddenSize, this->outputSize}, true);
            float* bias1Updated = arrayMulScalar(this->gradBias1, -learningRate, this->hiddenSize, true);
            float** weight1Updated = matMulScalar(this->weightGrad1, -learningRate, {this->inputSize, this->hiddenSize}, true);
            this->bias2 = arrayAddArray(this->bias2, bias2Updated, this->outputSize, false);
            this->weight2 = matAddMatrix(this->weight2, weight2Updated, {this->hiddenSize, this->outputSize}, {this->hiddenSize, this->outputSize}, false);
            this->bias1 = arrayAddArray(this->bias1, bias1Updated, this->hiddenSize, false);
            this->weight1 = matAddMatrix(this->weight1, weight1Updated, {this->inputSize, this->hiddenSize}, {this->inputSize, this->hiddenSize}, false);
            print1dArray(this->gradBias1, this->hiddenSize);
            print1dArray(bias1Updated, this->hiddenSize);
            print1dArray(this->bias1, this->hiddenSize);
            // Free Memory for linear2
            free2d(activatedHidden1Transposed, {this->hiddenSize, currBatchSize});
            free2d(weight2Updated, {this->hiddenSize, this->outputSize});
            free2d(proba, {currBatchSize, this->outputSize});
            free2d(grad1, {currBatchSize, this->outputSize});
            free2d(weight2Transposed, {this->outputSize, this->hiddenSize});
            free2d(grad2Activated1, {currBatchSize, this->hiddenSize});
            free2d(inputTransposed, {this->inputSize, currBatchSize});
            free2d(weight1Updated, {this->inputSize, this->hiddenSize});
            delete[] bias1Updated;
            delete[] bias2Updated;
            // Free all temp result and gradient
            this->freeTempResult();
            this->clearGrad();
        }
    }
}
