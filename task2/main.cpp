#include "./module.hpp"
#include "./fnn.hpp"
#include "./utils.cpp"

int main() {
    size_t inputSize = 200;
    size_t hiddenSize = 400;
    size_t outputSize = 500;
    size_t threadNum = 12;
    size_t perThreadDataSize = 500;
    std::cout << testPthread(inputSize, hiddenSize, outputSize, threadNum, perThreadDataSize, true) << std::endl;
    return 0;
}
