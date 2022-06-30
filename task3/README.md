# Parallel Task1: GPU

## Env

- NVIDIA Graphics Driver: 511.79

- CUDA Version: 10.1.243

## How to build

```
nvcc -o main main.cu
```

## Parameters

- `-d`: The number of data item, integer.

- `-f`: The feature size of input, integer.

- `-h`: The hidden size of neural network.

- `-o`: The output size of neural network.