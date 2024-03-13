#include <cuda_runtime.h>
// TODO: After multiple @dimOrder are fixed, generate ref and test entry
typedef float* mat89_f;
typedef int* mat89_i;

// dimOrder inside function argument
extern "C" __global__ void _occa_test_kernel_0_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat89_f mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            { ab[i] = a[i] + b[j] + mat[j + (9 * (i))]; }
            __syncthreads();
        }
    }
}

extern "C" __global__ void _occa_test_kernel_1_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat89_f mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            { ab[i] = a[i] + b[j] + mat[i + (8 * (j))]; }
            __syncthreads();
        }
    }
}

typedef float* mat98_f;

// typeDefs
extern "C" __global__ void _occa_test_kernel_2_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat98_f mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            { ab[i] = a[i] + b[j] + mat[mat[i + (9 * (j))] + (9 * (i))]; }
            __syncthreads();
        }
    }
}

// variable declaration
extern "C" __global__ void _occa_test_kernel_3_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat98_f mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            {
                ab[i] = a[i] + b[j] + mat[mat[i + (9 * (j))] + (9 * (i))];
                typedef int* mat23;
                mat23 xy;
                mat23 yx;
                yx[2 + (3 * (1))] = 0;
                xy[1 + (j * (2))] = yx[2 + (3 * (1))];
            }
            __syncthreads();
        }
    }
}
