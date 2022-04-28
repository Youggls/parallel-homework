#include <iostream>
#include "fnn.hpp"
#include "utils.hpp"
#include <ctime>

int main(int argc, char** argv) {
    // float** w = init2dArray(20, 30);
    // float* b = init1dArray(30);
    // float** x = init2dArray(10, 20);
    // linearLayer(x, w, b, 10, 20, 30, false);
    // Network* n = new Network(20, 30, 10, false);
    // n->forward(x, 10, false);
    bool simd = false;
    int dataSize = 1000;
    int featureSize = 100;
    int hiddenSize = 200;
    int outputSize = 200;
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-s") == 0) {
            if (strcmp(argv[i + 1], "true") == 0) {
                simd = true;
            } else if (strcmp(argv[i + 1], "false") == 0) {
                simd = false;
            } else {
                std::cerr << "Invalid argument for -s" << std::endl;
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
        }
    }
    Network* n = new Network(featureSize, hiddenSize, outputSize, simd);
    float** data = init2dArray(dataSize, featureSize);
    struct timespec start, end;
    std::cout << "Begin forward." << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    n->forward(data, dataSize);
    clock_gettime(CLOCK_MONOTONIC, &end);
    std::cout << "End forward." << std::endl;
    if (simd) {
        std::cout << "SIMD ";
    } else {
        std::cout << "Normal ";
    }
    std::cout << "time cost in total: " << (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000 << "ms" << std::endl;
    std::cout << "end" << std::endl;
    return 0;
}
