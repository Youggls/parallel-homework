#include "./module.hpp"
#include "./fnn.hpp"

int main() {
    size_t inputSize = 20;
    size_t hiddenSize = 40;
    size_t outputSize = 50;
    Network* n = new Network(inputSize, hiddenSize, outputSize, true);
    Matrix* input = new Matrix({40, inputSize});
    input->setOnes();
    Matrix* output = n->forward(input);
    return 0;
}
