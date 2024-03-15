#include <cuda_runtime.h>

extern "C" __global__ void _occa_function1_0(const int *data) {
  int i = (0) + blockIdx.x;
  {
    __shared__ int arr1[32];
    __shared__ float arr2[8][32];
    __shared__ double arr3[4 + 4];
    {
      int j = (0) + threadIdx.x;
      {}
    }
  }
}
