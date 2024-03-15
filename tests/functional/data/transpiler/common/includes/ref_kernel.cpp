#include <cuda_runtime.h>

struct Data {
    float* __restrict__ data;
    int* __restrict__ idxs;
};

__constant__ int SIZE = 128;
__device__ float add(float a, float b);

__device__ float add2(float a, float b) {
    return a + b;
}

__device__ inline float fastSqrt(float);

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_function1_0(const Data data1, const Data data2) {}
