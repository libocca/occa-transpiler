#include <cuda_runtime.h>

template <class T>
struct ComplexType {
    T real;
    T imaginary;
};

// TODO: fix me when @kernel/@outer/@inner will be implementeds
extern "C" __global__ void _occa_function1_0(const int* data) {
    int i = (0) + blockIdx.x;
    {
        __shared__ ComplexType<int> arr1[32];
        __shared__ ComplexType<float> arr2[8][32];
        {
            int j = (0) + threadIdx.x;
            {}
            __syncthreads();
        }
    }
}
