#include <cuda_runtime.h>

typedef struct {
    float* __restrict__ b;
    float* __restrict__ c;
} S;

float* __restrict__ aa;

extern "C" __global__ void _occa_hello_kern_0(S* a) {
    int i = (0) + blockIdx.x;
    {
        __shared__ float buf[100];
        int a, b;
        {
            int j = (0) + threadIdx.x;
            {
                a += 1;
                b += 1;
                __syncthreads();
            }
        }
    }
}
