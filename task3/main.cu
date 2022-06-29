#include <cuda_runtime.h>
#include <stdio.h>
#include "util.h"
#include <iostream>

//CPU对照组，用于对比加速比
void sumMatrix2DonCPU(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
    float* a = MatA;
    float* b = MatB;
    float* c = MatC;
    for(int j = 0; j < ny; j++) {
        for(int i = 0; i < nx; i++) {
            c[i] = a[i] + b[i];
        }
        c += nx;
        b += nx;
        a += nx;
    }
}

void matrixMulCpu(float* fpMatrixA, float* fpMatrixB, float* fpMatrixC, int m, int n, int k) {
    float sum = 0.0f;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int l = 0; l < k; l++) {
                sum += fpMatrixA[i * k + l] * fpMatrixB[l * n + j];
            }
            fpMatrixC[i * n + j] = sum;
            sum = 0.0f;
        }
    }
}

void arrayAddMatrixCpu(float* fpMatrixA, float* fpMatrixB, float* fpMatrixC, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            fpMatrixC[i * n + j] = fpMatrixA[i * n + j] + fpMatrixB[j];
        }
    }
}

//核函数，每一个线程计算矩阵中的一个元素。
__global__ void sumMatrix(float* MatA, float* MatB, float* MatC, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * ny;
    if (ix<nx && iy<ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

__global__ void arrayAddMatrix(const float* a, const float* b, float* c, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i + j * n;
    if (i < m && j < n) {
        c[idx]= a[idx] + b[j];
    }
}

__global__ void matrixAddMatrix(const float* a, const float* b, float* c, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i + j * n;
    if (i < m && j < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrixMulGlobalKernel(float* pfMatrixA, float* pfMatrixB, float* pfMatrixC, int m, int n, int k) {
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;
    for(int i = 0; i < k; i++) {
        fCVal += pfMatrixA[nRow * k + i] * pfMatrixB[i * n + nCol];
    }
    pfMatrixC[nRow * n + nCol] = fCVal;
}

void test(int batchSize, int featureSize ,int hiddenSize, int outSize) {
    printf("%d,%d,%d,", batchSize, featureSize, hiddenSize);
    int weight1NBytes = featureSize * hiddenSize * sizeof(float);
    int weight2NBytes = hiddenSize * outSize * sizeof(float);
    float* weight1 = (float*)malloc(weight1NBytes);
    float* weight2 = (float*)malloc(weight2NBytes);
    float* bias1 = (float*)malloc(hiddenSize * sizeof(float));
    float* bias2 = (float*)malloc(outSize * sizeof(float));
    initialData(weight1, featureSize * hiddenSize);
    initialData(weight2, hiddenSize * outSize);
    initialData(bias1, hiddenSize);
    initialData(bias2, outSize);
    float* weight1Dev = nullptr;
    float* weight2Dev = nullptr;
    float* bias1Dev = nullptr;
    float* bias2Dev = nullptr;
    CHECK(cudaMalloc((void**)&weight1Dev, weight1NBytes));
    CHECK(cudaMalloc((void**)&weight2Dev, weight2NBytes));
    CHECK(cudaMalloc((void**)&bias1Dev, hiddenSize * sizeof(float)));
    CHECK(cudaMalloc((void**)&bias2Dev, outSize * sizeof(float)));
    CHECK(cudaMemcpy(weight1Dev, weight1, weight1NBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(weight2Dev, weight2, weight2NBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(bias1Dev, bias1, hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(bias2Dev, bias2, outSize * sizeof(float), cudaMemcpyHostToDevice));
    int batchDataNBytes = batchSize * featureSize * sizeof(float);
    int tempDataNBytes = batchSize * hiddenSize * sizeof(float);
    int resultDataNBytes = batchSize * outSize * sizeof(float);
    float* batchData = (float*)malloc(batchDataNBytes);
    float* tempData = (float*)malloc(tempDataNBytes);
    float* resultData = (float*)malloc(resultDataNBytes);
    initialData(batchData, batchSize * featureSize);
    initialData(tempData, batchSize * hiddenSize);
    initialData(resultData, batchSize * outSize);
    float* batchDataDev = nullptr;
    float* tempDataDev = nullptr;
    float* resultDataDev = nullptr;
    CHECK(cudaMalloc((void**)&batchDataDev, batchDataNBytes));
    CHECK(cudaMalloc((void**)&tempDataDev, tempDataNBytes));
    CHECK(cudaMalloc((void**)&resultDataDev, resultDataNBytes));
    CHECK(cudaMemcpy(batchDataDev, batchData, batchDataNBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(tempDataDev, tempData, tempDataNBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(resultDataDev, resultData, resultDataNBytes, cudaMemcpyHostToDevice));
    dim3 block(32, 32);
    dim3 grid((batchSize - 1) / block.x + 1, (featureSize - 1) / block.y + 1);
    double gpuStart = cpuSecond();
    matrixMulGlobalKernel<<<grid, block>>>(batchDataDev, weight1Dev, tempDataDev, batchSize, featureSize, hiddenSize);
    arrayAddMatrix<<<grid, block>>>(tempDataDev, bias1Dev, tempDataDev, batchSize, hiddenSize);
    matrixMulGlobalKernel<<<grid, block>>>(tempDataDev, weight2Dev, resultDataDev, batchSize, hiddenSize, outSize);
    arrayAddMatrix<<<grid, block>>>(resultDataDev, bias2Dev, resultDataDev, batchSize, outSize);
    double gpuTime = cpuSecond() - gpuStart;
    cudaFree(batchDataDev);
    cudaFree(tempDataDev);
    cudaFree(resultDataDev);
    free(batchData);
    free(tempData);
    free(resultData);
    cudaFree(weight1Dev);
    cudaFree(weight2Dev);
    cudaFree(bias1Dev);
    cudaFree(bias2Dev);
    free(weight1);
    free(weight2);
    free(bias1);
    free(bias2);
    printf("%f\n", gpuTime * 1000);
}

int main(int argc, char** argv) {
    int batchSize = 64;
    int featureSize = 1024;
    int hiddenSize = 1024;
    int outSize = 32;
    for (size_t i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-d") == 0) {
            batchSize = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-f") == 0) {
            featureSize = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-h") == 0) {
            hiddenSize = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-o") == 0) {
            outSize = atoi(argv[i + 1]);
        } else {
            std::cout << argv[i];
            std::cerr << " Invalid argument" << std::endl;
            return 1;
        }
    }
    // printf("strating...\n");
    initDevice(0);
    test(batchSize, featureSize, hiddenSize, outSize);
    // // 输入二维矩阵，4096 * 4096，单精度浮点型。
    // int batchSizeArr[] = {512, 1024, 2048};
    // int featureSizeArr[] = {256, 512, 1024};
    // int hiddenSizeArr[] = {2048, 4096, 8196};
    // for (int i = 0; i < 3; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         for (int k = 0; k < 3; k++) {
    //             test(batchSizeArr[i], featureSizeArr[j], hiddenSizeArr[k], 32);
    //         }
    //     }
    // }
    // Malloc，开辟主机内存
    // float* A_host = (float*)malloc(nBytes);
    // float* B_host = (float*)malloc(nBytes);
    // float* C_host = (float*)malloc(nBytes);
    // float* C_from_gpu = (float*)malloc(nBytes);
    // initialData(A_host, nx * ny);
    // initialData(B_host, nx * ny);

    // // cudaMalloc，开辟设备内存
    // float* A_dev = NULL;
    // float* B_dev = NULL;
    // float* C_dev = NULL;
    // CHECK(cudaMalloc((void**)&A_dev, nBytes));
    // CHECK(cudaMalloc((void**)&B_dev, nBytes));
    // CHECK(cudaMalloc((void**)&C_dev, nBytes));

    // // 输入数据从主机内存拷贝到设备内存
    // CHECK(cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(B_dev, B_host, nBytes, cudaMemcpyHostToDevice));

    // dim3 block(32, 32);
    // dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);

    // // 测试 GPU 执行时间
    // double gpuStart = cpuSecond();
    // // 将核函数放在线程网格中执行
    // matrixAddMatrix<<<grid, block>>>(A_dev, B_dev, C_dev, nx, ny);
    // CHECK(cudaDeviceSynchronize());
    // double gpuTime = cpuSecond() - gpuStart;
    // printf("GPU Execution Time: %f sec\n", gpuTime);

    // // 在 CPU 上完成相同的任务
    // cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost);
    // double cpuStart = cpuSecond();
    // sumMatrix2DonCPU(A_host, B_host, C_host, nx, ny);
    // double cpuTime = cpuSecond() - cpuStart;
    // printf("CPU Execution Time: %f sec\n", cpuTime);

    // // 检查 GPU 与 CPU 计算结果是否相同
    // CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    // checkResult(C_host, C_from_gpu, nx * ny);

    // // 释放内存
    // cudaFree(A_dev);
    // cudaFree(B_dev);
    // cudaFree(C_dev);
    // free(A_host);
    // free(B_host);
    // free(C_host);
    // free(C_from_gpu);
    cudaDeviceReset();
    return 0;
}