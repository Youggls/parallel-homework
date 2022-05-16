#include "./module.hpp"
#include <random>
#include <iostream>
#ifdef __aarch64__
#include <arm_neon.h>
#elif __x86_64__
#include <pmmintrin.h>
#endif
using std::random_device;
using std::ranlux48;
using std::uniform_real_distribution;

bool Config::simd = false;

Matrix::Matrix(float** data, vector<size_t> shape) {
    this->data = data;
    this->shape = shape;
}

Matrix::Matrix(vector<size_t> shape) {
    this->shape = shape;
    this->data = new float*[this->shape[0]];
    for (size_t i = 0; i < this->shape[0]; i++) {
        this->data[i] = new float[this->shape[1]];
    }
}

Matrix::~Matrix() {
    for (size_t i = 0; i < this->shape[0]; i++) {
        delete[] this->data[i];
    }
    delete[] this->data;
}

void Matrix::random_init() {
    random_device seed;
    ranlux48 engine(seed());
    uniform_real_distribution<> distrib(-1, 1);
    for (size_t i = 0; i < this->shape[0]; i++) {
        for (size_t j = 0; j < this->shape[1]; j++) {
            this->data[i][j] = distrib(engine);
        }
    }
}

Matrix* Matrix::operator-() {
    for (size_t i = 0; i < this->shape[0]; i++) {
        for (size_t j = 0; j < this->shape[1]; j++) {
            this->data[i][j] = -this->data[i][j];
        }
    }
    return this;
}

Matrix* operator+(const Matrix& m1, const Matrix& m2) {
    if (m1.shape[0] != m2.shape[0] || m1.shape[1] != m2.shape[1]) {
        throw "Shape error!";
    }
    Matrix* result = new Matrix(m1.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = m1.data[i][j] + m2.data[i][j];
        }
    }
    return result;
}

Matrix* operator-(const Matrix& m1, const Matrix& m2) {
    if (m1.shape[0] != m2.shape[0] || m1.shape[1] != m2.shape[1]) {
        throw "Shape error!";
    }
    Matrix* result = new Matrix(m1.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = m1.data[i][j] - m2.data[i][j];
        }
    }
    return result;
}

Matrix* operator*(const Matrix& m1, const Matrix& m2) {
    if (m1.shape[1] != m2.shape[0]) {
        throw "Shape error!";
    }
    Matrix* result = new Matrix({m1.shape[0], m2.shape[1]});
    if (Config::simd) {
#ifdef __aarch64__
        for (size_t i = 0; i < m1.shape[0]; i += 1) {
            for (size_t j = 0; j < m2.shape[1]; j += 1) {
                for (size_t k = 0; k < m1.shape[1] - 3; k += 4) {
                    float32x4_t a4 = vld1q_f32(m1.data[i] + k);
                    float* tempB = new float[4];
                    for (size_t t = k; t < m1.shape[1] - 3; t += 4) {
                        tempB[t - k] = m2.data[t][j];
                    }
                    float32x4_t b4 = vld1q_f32(tempB);
                    float32x4_t c4 = vmulq_f32(a4, b4);
                    float32x2_t c2 = vadd_f32(vget_high_f32(c4), vget_low_f32(c4));
                    float32x2_t c1 = vpadd_f32(c2, c2);
                    result->data[i][j] += vget_lane_f32(c1, 0);
                }
                size_t mod = m1.shape[1] % 4;
                for (size_t k = m1.shape[1] - mod; k < m1.shape[1]; k++) {
                    result->data[i][j] += m1.data[i][k] * m2.data[k][j];
                }
            }
        }
#elif __x86_64__
        for (size_t i = 0; i < m1.shape[0]; i += 1) {
            for (size_t j = 0; j < m2.shape[1]; j += 1) {
                __m128 sum = _mm_setzero_ps();
                for (size_t k = 0; k < m1.shape[1] - 3; k += 4) {
                    __m128 t1 = _mm_loadu_ps(m1.data[i] + k);
                    float* tempB = new float[4];
                    for (size_t t = k; t < k + 4 && t < m1.shape[1] - 3; t += 1) {
                        tempB[t - k] = m2.data[t][j];
                    }
                    __m128 t2 = _mm_loadu_ps(tempB);
                    t1 = _mm_mul_ps(t1, t2);
                    sum = _mm_add_ps(sum, t1);
                }
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                _mm_store_ss(result->data[i] + j, sum);
                size_t mod = m1.shape[1] % 4;
                for (size_t k = m1.shape[1] - mod; k < m1.shape[1]; k++) {
                    result->data[i][j] += m1.data[i][k] * m2.data[k][j];
                }
            }
        }
#endif
    } else {
        for (size_t i = 0; i < m1.shape[0]; i += 1) {
            for (size_t j = 0; j < m2.shape[1]; j += 1) {
                float sum = 0.0;
                for (size_t k = 0; k < m1.shape[1]; k += 1) {
                    sum += m1.data[i][k] * m2.data[k][j];
                }
                result->data[i][j] = sum;
            }
        }
    }
    return result;
}

Matrix* operator+(const Matrix& m, const float salar) {
    Matrix* result = new Matrix(m.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = m.data[i][j] + salar;
        }
    }
    return result;
}

Matrix* operator+(const float salar, const Matrix& m) {
    Matrix* result = new Matrix(m.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = m.data[i][j] + salar;
        }
    }
    return result;
}

Matrix* operator-(const Matrix& m, const float salar) {
    Matrix* result = new Matrix(m.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = m.data[i][j] - salar;
        }
    }
    return result;
}

Matrix* operator-(const float salar, const Matrix& m) {
    Matrix* result = new Matrix(m.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = salar - m.data[i][j];
        }
    }
    return result;
}

Matrix* operator*(const Matrix& m, const float salar) {
    Matrix* result = new Matrix(m.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = m.data[i][j] * salar;
        }
    }
    return result;
}

Matrix* operator*(const float salar, const Matrix& m) {
    Matrix* result = new Matrix(m.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = salar * m.data[i][j];
        }
    }
    return result;
}

void Matrix::setValue(size_t i, size_t j, float value) {
    if (i >= this->shape[0] || j >= this->shape[1]) {
        throw "Index out of range!";
    }
    this->data[i][j] = value;
}

void Matrix::setAll(float value) {
    for (size_t i = 0; i < this->shape[0]; i += 1) {
        for (size_t j = 0; j < this->shape[1]; j += 1) {
            this->data[i][j] = value;
        }
    }
}

void Matrix::setOnes() {
    this->setAll(1.0);
}

void Matrix::setZeros() {
    this->setAll(0.0);
}

void Matrix::printMatrix() {
    for (size_t i = 0; i < this->shape[0]; i += 1) {
        for (size_t j = 0; j < this->shape[1]; j += 1) {
            std::cout << this->data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Array::Array(float* data, size_t shape) {
    this->data = data;
    this->shape = shape;
}

Array::Array(size_t shape) {
    this->data = new float[shape];
    this->shape = shape;
}

Array::~Array() {
    delete[] this->data;
}

void Array::random_init() {
    random_device seed;
    ranlux48 engine(seed());
    uniform_real_distribution<> distrib(-1, 1);
    for (size_t i = 0; i < this->shape; i++) {
        this->data[i] = distrib(engine);
    }
}

Array* Array::operator-() {
    Array* result = new Array(this->shape);
    for (size_t i = 0; i < this->shape; i++) {
        result->data[i] = -this->data[i];
    }
    return result;
}

Array* operator+(const Array& a1, const Array& a2) {
    if (a1.shape != a2.shape) {
        throw "Shape error!";
    }
    Array* result = new Array(a1.shape);
    for (size_t i = 0; i < a1.shape; i++) {
        result->data[i] = a1.data[i] + a2.data[i];
    }
    return result;
}

Array* operator-(const Array& a1, const Array& a2) {
    if (a1.shape != a2.shape) {
        throw "Shape error!";
    }
    Array* result = new Array(a1.shape);
    for (size_t i = 0; i < a1.shape; i++) {
        result->data[i] = a1.data[i] - a2.data[i];
    }
    return result;
}

Matrix* operator+(const Array& a, const Matrix& m) {
    if (a.shape != m.shape[1]) {
        throw "Shape error!";
    }
    Matrix* result = new Matrix(m.shape);
    for (size_t j = 0; j < result->shape[0]; j += 1) {
        for (size_t i = 0; i < result->shape[1]; i += 1) {
            result->data[i][j] = m.data[i][j] + a.data[j];
        }
    }
    return result;
}

Matrix* operator+(const Matrix& m, const Array& a) {
    if (a.shape != m.shape[1]) {
        throw "Shape error!";
    }
    Matrix* result = new Matrix(m.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        if (Config::simd) {
#ifdef __aarch64__
            for (size_t j = 0; j < result->shape[1] - 3; j += 4) {
                float32x4_t a4 = vld1q_f32(a[i] + j);
                float32x4_t b4 = vld1q_f32(b + j);
                float32x4_t c4 = vaddq_f32(a4, b4);
                vst1q_f32(c[i] + j, c4);
            }
            size_t mod = result->shape[1] % 4;
            for (size_t j = result->shape[1] - mod; j < result->shape[1]; j++) {
                c[i][j] = a[i][j] + b[j];
            }
#elif __x86_64__
            for (size_t j = 0; j < result->shape[1] - 3; j += 4) {
                __m128 t1 = _mm_loadu_ps(m.data[i] + j);
                __m128 t2 = _mm_loadu_ps(a.data + j);
                t1 = _mm_add_ps(t1, t2);
                _mm_storeu_ps(result->data[i] + j, t1);
            }
            size_t mod = result->shape[1] % 4;
            for (size_t j = result->shape[1] - mod; j < result->shape[1]; j++) {
                result->data[i][j] = m.data[i][j] + a.data[j];
            }
#endif
        } else {
            for (size_t j = 0; j < result->shape[1]; j += 1) {
                result->data[i][j] = m.data[i][j] + a.data[j];
            }
        }
    }
    return result;
}

void Array::setValue(size_t i, float value) {
    if (i >= this->shape) {
        throw "Index out of range!";
    }
    this->data[i] = value;
}

void Array::setAll(float value) {
    for (size_t i = 0; i < this->shape; i += 1) {
        this->data[i] = value;
    }
}

void Array::setOnes() {
    this->setAll(1.0);
}

void Array::setZeros() {
    this->setAll(0.0);
}

void Array::printArray() {
    for (size_t i = 0; i < this->shape; i += 1) {
        std::cout << this->data[i] << " ";
    }
    std::cout << std::endl;
}
