#include "./module.hpp"

int main() {
    Matrix* a = new Matrix({2, 3});
    a->setOnes();
    Matrix* b = new Matrix({3, 2});
    b->setOnes();
    Matrix* c = (*a) + 2;
    Matrix* d = -(*a);
    d->printMatrix();
    Array* arr = new Array(2);
    arr->setOnes();
    arr = *arr + *arr;
    arr->printArray();
    Matrix* test = *a + *arr;
    test->printMatrix();
    return 0;
}
