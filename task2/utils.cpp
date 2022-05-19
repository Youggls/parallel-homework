#include "fnn.hpp"
#include "module.hpp"
#include <pthread.h>
#include <iostream>
#include <ctime>

struct ThreadData
{
    Matrix* data;
    Network* net;
};


void* thread_worker(void* arg) {
    struct ThreadData* threadData = (struct ThreadData*) arg;
    Matrix* data = threadData->data;
    Network* net = threadData->net;
    net->forward(data);
}

int testPthread(size_t inputSize, size_t hiddenSize, size_t outputSize, size_t threadNum, size_t perThreadDataSize, bool simd) {
    std::cout << "Test Pthread" << std::endl;
    std::cout << "Input Size: " << inputSize << std::endl;
    std::cout << "Hidden Size: " << hiddenSize << std::endl;
    std::cout << "Output Size: " << outputSize << std::endl;
    std::cout << "Thread Number: " << threadNum << std::endl;
    std::cout << "Per Thread Data Size: " << perThreadDataSize << std::endl;
    std::cout << "Total Data Size: " << perThreadDataSize * threadNum << std::endl;
    Network** nets = new Network*[threadNum];
    Matrix** mats = new Matrix*[threadNum];
    struct ThreadData** threadDatas = new struct ThreadData*[threadNum];
    pthread_t* threads = new pthread_t[threadNum];    
    for (size_t i = 0; i < threadNum; i++) {
        nets[i] = new Network(inputSize, hiddenSize, outputSize, simd);
        mats[i] = new Matrix({perThreadDataSize, inputSize});
        threadDatas[i] = new struct ThreadData;
        threadDatas[i]->data = mats[i];
        threadDatas[i]->net = nets[i];
        pthread_create(&threads[i], NULL, thread_worker, (void*)threadDatas[i]);
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (size_t i = 0; i < threadNum; i++) {
        void* thrd_ret;
        int res = pthread_join(threads[i], &thrd_ret);
        if (!res) {
            printf("Thread %ld joined\n", i);
        } else {
            printf("Thread %ld join failed\n", i);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
}
