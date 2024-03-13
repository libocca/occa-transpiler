#include <cuda_runtime.h>

struct ComplexValueFloat {
    float real;
    float imaginary;
};

// TODO: fix me when @kernel/@outer/@inner will be implementeds
extern "C" __global__ void _occa_function1_0(const int* data) {
    __shared__ ComplexValueFloat arr1[32];
    __shared__ ComplexValueFloat arr2[8][32];
}
