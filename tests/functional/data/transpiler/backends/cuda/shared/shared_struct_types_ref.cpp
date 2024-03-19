#include <cuda_runtime.h>

struct ComplexValueFloat {
  float real;
  float imaginary;
};

extern "C" __global__ void _occa_function1_0(const int *data) {
  {
    int i = (0) + blockIdx.x;
    __shared__ ComplexValueFloat arr2[8][32];
    __shared__ ComplexValueFloat arr1[32];
    {
      int j = (0) + threadIdx.x;
    }
  }
}
