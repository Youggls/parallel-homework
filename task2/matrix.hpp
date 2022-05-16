#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <vector>
using std::vector;

class Matrix {
private:
    float** data;
    vector<size_t> shape;
    void random_init();
public:
    Matrix(float** data, vector<size_t> shape);
    Matrix(vector<size_t> shape);
    Matrix& operator-();
    friend Matrix& operator+(const Matrix& m1, const Matrix& m2);
    friend Matrix& operator-(const Matrix& m1, const Matrix& m2);
    friend Matrix& operator*(const Matrix& m1, const Matrix& m2);
    friend Matrix& operator+(const Matrix& m, const float salar);
    friend Matrix& operator+(const float salar, const Matrix& m);
    friend Matrix& operator-(const Matrix& m, const float salar);
    friend Matrix& operator-(const float salar, const Matrix& m);
    friend Matrix& operator*(const Matrix& m, const float salar);
    friend Matrix& operator*(const float salar, const Matrix& m);
};

#endif