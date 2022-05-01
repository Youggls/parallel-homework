#include <arm_neon.h>
#include <vector>
#include <ctime>
#include <iostream>
using std::vector;

float** matmul(float** a, float** b, float** c, vector<size_t> shapeA, vector<size_t> shapeB) {
    if (shapeA[1] != shapeB[0]) {
        throw "matmul: shape mismatch";
    }
    for (size_t i = 0; i < shapeA[0]; i += 1) {
        for (size_t j = 0; j < shapeB[1]; j += 1) {
            for (size_t k = 0; k < shapeA[1] - 3; k += 4) {
                float32x4_t a4 = vld1q_f32(a[i] + k);
                float32x4_t b4 = vld1q_f32(b[j] + k);
                float32x4_t c4 = vmulq_f32(a4, b4);
                float32x2_t c2 = vadd_f32(vget_high_f32(c4), vget_low_f32(c4));
                float32x2_t c1 = vpadd_f32(c2, c2);
                c[i][j] += vget_lane_f32(c1, 0);
            }
            size_t mod = shapeA[1] % 4;
            for (size_t k = shapeA[1] - mod; k < shapeA[1]; k++) {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    return c;
}

float** matmulNormal(float** a, float** b, float** c, vector<size_t> shapeA, vector<size_t> shapeB) {
    if (shapeA[1] != shapeB[0]) {
        throw "matmul: shape mismatch";
    }
    for (size_t i = 0; i < shapeA[0]; i += 1) {
        for (size_t j = 0; j < shapeB[1]; j += 1) {
            for (size_t k = 0; k < shapeA[1]; k += 1) {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    return c;
}

int main() {
    float** a = new float*[1024];
    float** b = new float*[1024];
    float** c = new float*[1024];
    for (size_t i = 0; i < 1024; i += 1) {
        a[i] = new float[1024];
        b[i] = new float[1024];
        c[i] = new float[1024];
    }
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul(a, b, c, {1024, 1024}, {1024, 1024});
    clock_gettime(CLOCK_MONOTONIC, &end);
    std::cout << "matmul time cost in total: " << (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000 << "ms" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmulNormal(a, b, c, {1024, 1024}, {1024, 1024});
    clock_gettime(CLOCK_MONOTONIC, &end);
    std::cout << "matmulNormal time cost in total: " << (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000 << "ms" << std::endl;
    return 0;
}
