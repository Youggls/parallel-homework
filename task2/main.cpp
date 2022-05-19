#include "./module.hpp"
#include "./fnn.hpp"
#include "./utils.cpp"

int main() {
    size_t inputSize = 200;
    size_t hiddenSize = 400;
    size_t outputSize = 500;
    size_t threadNum = 12;
    size_t perThreadDataSize = 500;
    vector<size_t> threadNums = {1, 2, 4, 8, 16, 32};
    vector<size_t> perThreadDataSizes = {4096, 2048, 1024, 512, 256, 128};
    for (int i = 0; i < threadNums.size(); i++) {
        size_t threadNum = threadNums[i];
        size_t perThreadDataSize = perThreadDataSizes[i];
        testPthread(inputSize, hiddenSize, outputSize, threadNum, perThreadDataSize, false);
    }
    return 0;
}
