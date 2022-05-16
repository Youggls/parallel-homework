#include "./matrix.hpp"
#include <random>
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

Matrix& Matrix::operator-() {
    for (size_t i = 0; i < this->shape[0]; i++) {
        for (size_t j = 0; j < this->shape[1]; j++) {
            this->data[i][j] = -this->data[i][j];
        }
    }
    return *this;
}

Matrix& operator+(const Matrix& m1, const Matrix& m2) {
    if (m1.shape[0] != m2.shape[0] || m1.shape[1] != m2.shape[1]) {
        throw "Shape error!";
    }
    Matrix result(m1.shape);
    for (size_t i = 0; i < result.shape[0]; i += 1) {
        for (size_t j = 0; j < result.shape[1]; j += 1) {
            result.data[i][j] = m1.data[i][j] + m2.data[i][j];
        }
    }
    return result;
}

Matrix& operator-(const Matrix& m1, const Matrix& m2) {
    if (m1.shape[0] != m2.shape[0] || m1.shape[1] != m2.shape[1]) {
        throw "Shape error!";
    }
    Matrix result(m1.shape);
    for (size_t i = 0; i < result.shape[0]; i += 1) {
        for (size_t j = 0; j < result.shape[1]; j += 1) {
            result.data[i][j] = m1.data[i][j] - m2.data[i][j];
        }
    }
    return result;
}

Matrix& operator*(const Matrix& m1, const Matrix& m2) {
    if (m1.shape[1] != m2.shape[0]) {
        throw "Shape error!";
    }
    Matrix result(m1.shape);
    for (size_t i = 0; i < m1.shape[0]; i += 1) {
        for (size_t j = 0; j < m1.shape[1]; j += 1) {
            float sum = 0.0;
            for (size_t k = 0; k < m2.shape[0]; k += 1) {
                sum += m1.data[i][k] * m2.data[k][j];
            }
        }
    }
    return result;
}

Matrix& operator+(const Matrix& m, const float salar) {
    Matrix result(m.shape);
    for (size_t i = 0; i < result.shape[0]; i += 1) {
        for (size_t j = 0; j < result.shape[1]; j += 1) {
            result.data[i][j] = m.data[i][j] + salar;
        }
    }
    return result;
}

Matrix& operator+(const float salar, const Matrix& m) {
    Matrix result(m.shape);
    for (size_t i = 0; i < result.shape[0]; i += 1) {
        for (size_t j = 0; j < result.shape[1]; j += 1) {
            result.data[i][j] = m.data[i][j] + salar;
        }
    }
    return result;
}

Matrix& operator-(const Matrix& m, const float salar) {
    Matrix result(m.shape);
    for (size_t i = 0; i < result.shape[0]; i += 1) {
        for (size_t j = 0; j < result.shape[1]; j += 1) {
            result.data[i][j] = m.data[i][j] - salar;
        }
    }
    return result;
}

Matrix& operator-(const float salar, const Matrix& m) {
    Matrix result(m.shape);
    for (size_t i = 0; i < result.shape[0]; i += 1) {
        for (size_t j = 0; j < result.shape[1]; j += 1) {
            result.data[i][j] = salar - m.data[i][j];
        }
    }
    return result;
}

Matrix& operator*(const Matrix& m, const float salar) {
    Matrix result(m.shape);
    for (size_t i = 0; i < result.shape[0]; i += 1) {
        for (size_t j = 0; j < result.shape[1]; j += 1) {
            result.data[i][j] = m.data[i][j] * salar;
        }
    }
    return result;
}

Matrix& operator*(const float salar, const Matrix& m) {
    Matrix result(m.shape);
    for (size_t i = 0; i < result.shape[0]; i += 1) {
        for (size_t j = 0; j < result.shape[1]; j += 1) {
            result.data[i][j] = salar * m.data[i][j];
        }
    }
    return result;
}
