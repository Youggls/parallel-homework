#include "./module.hpp"
#include <random>
#include <iostream>
using std::random_device;
using std::ranlux48;
using std::uniform_real_distribution;

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
    for (size_t i = 0; i < m1.shape[0]; i += 1) {
        for (size_t j = 0; j < m1.shape[1]; j += 1) {
            float sum = 0.0;
            for (size_t k = 0; k < m2.shape[0]; k += 1) {
                sum += m1.data[i][k] * m2.data[k][j];
            }
            result->data[i][j] = sum;
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
    if (a.shape != m.shape[0]) {
        throw "Shape error!";
    }
    Matrix* result = new Matrix(m.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = a.data[i] + m.data[i][j];
        }
    }
    return result;
}

Matrix* operator+(const Matrix& m, const Array& a) {
    if (a.shape != m.shape[0]) {
        throw "Shape error!";
    }
    Matrix* result = new Matrix(m.shape);
    for (size_t i = 0; i < result->shape[0]; i += 1) {
        for (size_t j = 0; j < result->shape[1]; j += 1) {
            result->data[i][j] = m.data[i][j] + a.data[j];
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
