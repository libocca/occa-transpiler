#include <cuda_runtime.h>

// TODO: fix me when @kernel/@outer/@inner will be implementeds
extern "C" __global__ void _occa_function1_0(const int* data) {
    __shared__ int arr1[32];
    __shared__ float arr2[8][32];
    __shared__ double arr3[4 + 4];
}
