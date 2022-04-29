#include "utils.hpp"
#include <bits/stdc++.h>
#ifdef __aarch64__
#include <arm_neon.h>
#elif __x86_64__
#include <pmmintrin.h>
#endif
using std::vector;

void free2d(float** a, vector<size_t> shape) {
    for (size_t i = 0; i < shape[0]; i++) {
        delete[] a[i];
    }
    delete[] a;
    a = nullptr;
}

float* init1dArray(size_t size, float defaultValue) {
    float* a = new float[size];
    memset(a, defaultValue, sizeof(float));
    return a;
}

void print1dArray(float* a, size_t size) {
    for (size_t i = 0; i < size; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

void print2dArray(float** a, vector<size_t> shape) {
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }
}

float** init2dArray(size_t m, size_t n, float defaultValue) {
    float** a = new float*[m];
    for (size_t i = 0; i < m; i++) {
        a[i] = new float[n];
        memset(a[i], defaultValue, sizeof(float) * n);
    }
    return a;
}

float** matmul(float** a, float** b, vector<size_t> shapeA, vector<size_t> shapeB, bool simd, bool cache) {
    if (shapeA[1] != shapeB[0]) {
        throw "matmul: shape mismatch";
    }
    float** c = init2dArray(shapeA[0], shapeB[1]);
    float** bT = nullptr;
    if (simd || cache) {
        bT = transpose(b, shapeB);
    }
    if (simd) {
#ifdef __aarch64__
        for (size_t i = 0; i < shapeA[0]; i += 1) {
            for (size_t j = 0; j < shapeB[1]; j += 1) {
                for (size_t k = 0; k < shapeA[1] - 3; k += 4) {
                    float32x4_t a4 = vld1q_f32(a[i] + k);
                    float32x4_t b4 = vld1q_f32(bT[j] + k);
                    float32x4_t c4 = vmulq_f32(a4, b4);
                    float32x2_t c2 = vadd_f32(vget_high_f32(c4), vget_low_f32(c4));
                    float32x2_t c1 = vpadd_f32(c2, c2);
                    c[i][j] += vget_lane_f32(c1, 0);
                }
                size_t mod = shapeA[1] % 4;
                for (size_t k = shapeA[1] - mod; k < shapeA[1]; k++) {
                    c[i][j] += a[i][k] * bT[j][k];
                }
            }
        }
#elif __x86_64__
        for (size_t i = 0; i < shapeA[0]; i += 1) {
            for (size_t j = 0; j < shapeB[1]; j += 1) {
                __m128 sum = _mm_setzero_ps();
                for (size_t k = 0; k < shapeA[1] - 3; k += 4) {
                    __m128 t1 = _mm_loadu_ps(a[i] + k);
                    __m128 t2 = _mm_loadu_ps(bT[j] + k);
                    t1 = _mm_mul_ps(t1, t2);
                    sum = _mm_add_ps(sum, t1);
                }
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                _mm_store_ss(c[i] + j, sum);
                size_t mod = shapeA[1] % 4;
                for (size_t k = shapeA[1] - mod; k < shapeA[1]; k++) {
                    c[i][j] += a[i][k] * bT[j][k];
                }
            }
        }
#else
        for (size_t i = 0; i < shapeA[0]; i++) {
            for (size_t j = 0; j < shapeB[1]; j++) {
                for (size_t k = 0; k < shapeA[1]; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
#endif
    }
    else {
        if (cache) {
            for (size_t i = 0; i < shapeA[0]; i++) {
                for (size_t j = 0; j < shapeB[1]; j++) {
                    for (size_t k = 0; k < shapeA[1]; k++) {
                        c[i][j] += a[i][k] * bT[j][k];
                    }
                }
            }
        } else {
            for (size_t i = 0; i < shapeA[0]; i++) {
                for (size_t j = 0; j < shapeB[1]; j++) {
                    for (size_t k = 0; k < shapeA[1]; k++) {
                        c[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        }
    }
    if (simd || cache) {
        free2d(bT, {shapeB[1], shapeB[0]});
    }
    return c;
}

float** matMulElement(float** a, float** b, vector<size_t> shapeA, vector<size_t> shapeB, bool generateNewCopy) {
    float** c = generateNewCopy? init2dArray(shapeA[0], shapeA[1]) : a;
    for (size_t i = 0; i < shapeA[0]; i++) {
        for (size_t j = 0; j < shapeB[1]; j++) {
            c[i][j] = a[i][j] * b[i][j];
        }
    }
    return c;
}

float** matAdd1dArray(float** a, float* b, vector<size_t> shapeA, vector<size_t> shapeB, bool generateNewCopy, bool simd) {
    if (shapeA[1] != shapeB[0]) {
        throw "matAdd1dArray: shape mismatch";
    }
    float** c = generateNewCopy ? init2dArray(shapeA[0], shapeA[1], 0.0) : a;
    for (size_t i = 0; i < shapeA[0]; i++) {
        if (simd) {
#ifdef __aarch64__
            for (size_t j = 0; j < shapeA[1] - 3; j += 4) {
                float32x4_t a4 = vld1q_f32(a[i] + j);
                float32x4_t b4 = vld1q_f32(b + j);
                float32x4_t c4 = vaddq_f32(a4, b4);
                vst1q_f32(c[i] + j, c4);
            }
            size_t mod = shapeA[1] % 4;
            for (size_t j = shapeA[1] - mod; j < shapeA[1]; j++) {
                c[i][j] = a[i][j] + b[j];
            }
#elif __x86_64__
            for (size_t j = 0; j < shapeA[1] - 3; j += 4) {
                __m128 t1 = _mm_loadu_ps(a[i] + j);
                __m128 t2 = _mm_loadu_ps(b + j);
                t1 = _mm_add_ps(t1, t2);
                _mm_storeu_ps(c[i] + j, t1);
            }
            size_t mod = shapeA[1] % 4;
            for (size_t j = shapeA[1] - mod; j < shapeA[1]; j++) {
                c[i][j] = a[i][j] + b[j];
            }
#else
            for (size_t j = 0; j < shapeA[1]; j++) {
                c[i][j] = a[i][j] + b[j];
            }
#endif
        }
        else {
            for (size_t j = 0; j < shapeA[1]; j++) {
                c[i][j] = a[i][j] + b[j];
            }
        }
    }
    return c;
}

float** matAddMatrix(float** a, float** b, vector<size_t> shapeA, vector<size_t> shapeB, bool generateNewCopy) {
    if (shapeA[0] != shapeB[0] || shapeA[1] != shapeB[1]) {
        throw "matAddMatrix: shape mismatch";
    }
    float** c = generateNewCopy ? init2dArray(shapeA[0], shapeA[1], 0.0) : a;
    for (size_t i = 0; i < shapeA[0]; i++) {
        for (size_t j = 0; j < shapeA[1]; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}

float** matMinusScalar(float** a, float b, vector<size_t> shapeA, bool generateNewCopy) {
    float** c = generateNewCopy ? init2dArray(shapeA[0], shapeA[1]) : a;
    for (size_t i = 0; i < shapeA[0]; i++) {
        for (size_t j = 0; j < shapeA[1]; j++) {
            c[i][j] = a[i][j] - b;
        }
    }
    return c;
}

float** matMulScalar(float** a, float b, vector<size_t> shapeA, bool generateNewCopy) {
    float** c = generateNewCopy ? init2dArray(shapeA[0], shapeA[1]) : a;
    for (size_t i = 0; i < shapeA[0]; i++) {
        for (size_t j = 0; j < shapeA[1]; j++) {
            c[i][j] = a[i][j] * b;
        }
    }
    return c;
}

float** matMinusMat(float** a, float** b, vector<size_t> shape, bool generateNewCopy) {
    float** c = generateNewCopy ? init2dArray(shape[0], shape[1]) : a;
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    return c;
}

float* arrayAddScalar(float* a, float b, size_t size, bool generateNewCopy) {
    float* c = generateNewCopy ? init1dArray(size, 0.0) : a;
    for (size_t i = 0; i < size; i++) {
        c[i] = a[i] + b;
    }
    return c;
}

float* arrayMulScalar(float* a, float b, size_t size, bool generateNewCopy) {
    float* c = generateNewCopy ? init1dArray(size, 0.0) : a;
    for (size_t i = 0; i < size; i++) {
        c[i] = a[i] * b;
    }
    return c;
}

float* arrayAddArray(float* a, float* b, size_t size, bool generateNewCopy) {
    float* c = generateNewCopy ? init1dArray(size, 0.0) : a;
    for (size_t i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}

float** linearLayer(float** input, float** weight, float* bias, size_t batchSize, size_t inputSize, size_t outputSize, bool simd, bool cache) {
    vector<size_t> inputShape = {batchSize, inputSize};
    vector<size_t> weightShape = {inputSize, outputSize};
    vector<size_t> outputShape = {batchSize, outputSize};
    float** output = matmul(input, weight, inputShape, weightShape, simd, cache);
    // Wouldn't re allocating memory
    output = matAdd1dArray(output, bias, outputShape, {outputSize}, true, simd);
    return output;
}


float** softmax(float** a, vector<size_t> shapeA) {
    float** c = init2dArray(shapeA[0], shapeA[1]);
    float sum = 0;
    for (size_t i = 0; i < shapeA[0]; i++) {
        float sum = 0;
        for (size_t j = 0; j < shapeA[1]; j++) {
            sum += exp(a[i][j]);
        }
        for (size_t j = 0; j < shapeA[1]; j++) {
            c[i][j] = exp(a[i][j]) / sum;
        }
    }
    return c;
}

float** sigmoid(float** a, vector<size_t> shapeA) {
    float** c = init2dArray(shapeA[0], shapeA[1]);
    for (size_t i = 0; i < shapeA[0]; i++) {
        for (size_t j = 0; j < shapeA[1]; j++) {
            c[i][j] = 1 / (1 + exp(-a[i][j]));
        }
    }
    return c;
}

float** sigmoidDerivative(float** a, vector<size_t> shapeA) {
    float** c = init2dArray(shapeA[0], shapeA[1]);
    for (size_t i = 0; i < shapeA[0]; i++) {
        for (size_t j = 0; j < shapeA[1]; j++) {
            c[i][j] = a[i][j] * (1 - a[i][j]);
        }
    }
    return c;
}

float** transpose(float** a, vector<size_t> shapeA) {
    float** c = init2dArray(shapeA[1], shapeA[0]);
    for (size_t i = 0; i < shapeA[0]; i++) {
        for (size_t j = 0; j < shapeA[1]; j++) {
            c[j][i] = a[i][j];
        }
    }
    return c;
}

float* sum2dLine(float** a, vector<size_t> shapeA) {
    float* sum = init1dArray(shapeA[0], 0);
    for (size_t i = 0; i < shapeA[0]; i++) {
        for (size_t j = 0; j < shapeA[1]; j++) {
            sum[i] += a[i][j];
        }
    }
    return sum;
}

float* sum2dCol(float** a, vector<size_t> shapeA) {
    float* sum = init1dArray(shapeA[1], 0);
    for (size_t i = 0; i < shapeA[0]; i++) {
        for (size_t j = 0; j < shapeA[1]; j++) {
            sum[j] += a[i][j];
        }
    }
    return sum;
}
