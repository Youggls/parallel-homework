#include <iostream>
#include "fnn.hpp"
#include "utils.hpp"
#include <ctime>

void printSIMD(bool simd) {
    if (simd) {
        std::cout << "SIMD ";
    } else {
        std::cout << "Normal ";
    }
}

void testAll(size_t dataSize, size_t featureSize, size_t hiddenSize, size_t outputSize, bool simd, bool cache) {
    Network* n = new Network(featureSize, hiddenSize, outputSize, simd, cache);
    float** data = init2dArray(dataSize, featureSize);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    n->forward(data, dataSize, false);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printSIMD(simd);
    std::cout << "NN forward time cost in total: " << (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000 << "ms" << std::endl;
    std::cout << "end" << std::endl;
}

void testMatMul(size_t dataSize, size_t featureSize, size_t hiddenSize, size_t outputSize, bool simd, bool cache) {
    struct timespec start, end;
    float** data = init2dArray(dataSize, featureSize);
    float** weight1 = init2dArray(featureSize, hiddenSize);
    float** weight2 = init2dArray(hiddenSize, outputSize);
    clock_gettime(CLOCK_MONOTONIC, &start);
    float** res1 = matmul(data, weight1, {dataSize, featureSize}, {featureSize, hiddenSize}, simd, cache);
    float** res2 = matmul(res1, weight2, {dataSize, hiddenSize}, {hiddenSize, outputSize}, simd, cache);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printSIMD(simd);
    std::cout << "matrix multiply time cost in total: " << (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000 << "ms" << std::endl;
}

void testMatAdd(size_t dataSize, size_t featureSize, size_t hiddenSize, size_t outputSize, bool simd) {
    struct timespec start, end;
    float** h1 = init2dArray(dataSize, hiddenSize);
    float** h2 = init2dArray(dataSize, outputSize);
    float* bias1 = init1dArray(hiddenSize);
    float* bias2 = init1dArray(outputSize);
    clock_gettime(CLOCK_MONOTONIC, &start);
    float** res1 = matAdd1dArray(h1, bias1, {dataSize, hiddenSize}, {hiddenSize}, simd);
    float** res2 = matAdd1dArray(h2, bias2, {dataSize, outputSize}, {outputSize}, simd);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printSIMD(simd);
    std::cout << "matrix add time cost in total: " << (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000 << "ms" << std::endl;
}

int main(int argc, char** argv) {
    bool simd = false;
    bool cache = false;
    size_t dataSize = 1000;
    size_t featureSize = 100;
    size_t hiddenSize = 200;
    size_t outputSize = 200;
    // 0: testAll
    // 1: testMatMul
    // 2: testMatAdd
    int task = 0;
    for (size_t i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-s") == 0) {
            if (strcmp(argv[i + 1], "true") == 0) {
                simd = true;
            } else if (strcmp(argv[i + 1], "false") == 0) {
                simd = false;
            } else {
                std::cerr << "Invalid argument for -s" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-c") == 0) {
            if (strcmp(argv[i + 1], "true") == 0) {
                cache = true;
            } else if (strcmp(argv[i + 1], "false") == 0) {
                cache = false;
            } else {
                std::cerr << "Invalid argument for -c" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0) {
            dataSize = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-f") == 0) {
            featureSize = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-h") == 0) {
            hiddenSize = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-o") == 0) {
            outputSize = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-t") == 0) {
            task = atoi(argv[i + 1]);
        } else {
            std::cout << argv[i];
            std::cerr << " Invalid argument" << std::endl;
            return 1;
        }
    }
    std::cout << "******** Configuration *********" << std::endl;
    std::cout << "dataSize: " << dataSize << std::endl;
    std::cout << "featureSize: " << featureSize << std::endl;
    std::cout << "hiddenSize: " << hiddenSize << std::endl;
    std::cout << "outputSize: " << outputSize << std::endl;
    std::cout << "SIMD optimize: " << simd << std::endl;
    std::cout << "Cache optimize: " << cache << std::endl;
    if (task == 0) {
        testAll(dataSize, featureSize, hiddenSize, outputSize, simd, cache);
    } else if (task == 1) {
        testMatMul(dataSize, featureSize, hiddenSize, outputSize, simd, cache);
    } else if (task == 2) {
        testMatAdd(dataSize, featureSize, hiddenSize, outputSize, simd);
    } else {
        std::cerr << "Invalid task" << std::endl;
        return 1;
    }
    
    return 0;
}
