# Parallel Task1: SIMD

## How to build

### ARM

- ARM Device:

    ```shell
    make build-arm
    ```

- QEMU (X86 Device):

    ```shell
    make build-qemu
    ```

### X86

```
make build
```

## Parameters

- `-s`: Whether to use SIMD optimize, `true` or `false`.

- `-d`: The number of data item, integer.

- `-c`: Whether to use cache optimize, `true` or `false`.

- `-f`: The feature size of input, integer.

- `-h`: The hidden size of neural network.

- `-o`: The output size of neural network.

- `-t`: The task id. `0` means test all function. `1` means only test matrix multiply. `2` means only test matrix add vector. `3` means test memory allocate and free.
