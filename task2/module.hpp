#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <vector>
using std::vector;
class Array;

class Config {
public:
    static bool simd;
    static void setSimd(bool simd) {
        Config::simd = simd;
    }
};

class Matrix {
private:
    float** data;
    vector<size_t> shape;
    void random_init();
public:
    Matrix(float** data, vector<size_t> shape);
    Matrix(vector<size_t> shape);
    ~Matrix();
    Matrix* operator-();
    friend Matrix* operator+(const Matrix& m1, const Matrix& m2);
    friend Matrix* operator-(const Matrix& m1, const Matrix& m2);
    friend Matrix* operator*(const Matrix& m1, const Matrix& m2);
    friend Matrix* operator+(const Matrix& m, const float salar);
    friend Matrix* operator+(const float salar, const Matrix& m);
    friend Matrix* operator-(const Matrix& m, const float salar);
    friend Matrix* operator-(const float salar, const Matrix& m);
    friend Matrix* operator*(const Matrix& m, const float salar);
    friend Matrix* operator*(const float salar, const Matrix& m);
    friend Matrix* operator+(const Array& a, const Matrix& m);
    friend Matrix* operator+(const Matrix& m, const Array& a);
    void setValue(size_t i, size_t j, float value);
    void setAll(float value);
    void setOnes();
    void setZeros();
    void printMatrix();
};

class Array {
private:
    float* data;
    size_t shape;
    void random_init();
public:
    Array(float* data, size_t shape);
    Array(size_t shape);
    ~Array();
    Array* operator-();
    friend Array* operator+(const Array& a1, const Array& a2);
    friend Array* operator-(const Array& a1, const Array& a2);
    friend Matrix* operator+(const Array& a, const Matrix& m);
    friend Matrix* operator+(const Matrix& m, const Array& a);

    void setValue(size_t i, float value);
    void setAll(float value);
    void setOnes();
    void setZeros();
    void printArray();
};

#endif