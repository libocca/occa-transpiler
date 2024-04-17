#include <cuda_runtime.h>

extern "C" __global__ __launch_bounds__(100) void _occa_buggy_kernel_0() {
  {
    int i = (0) + blockIdx.x;
    __shared__ float shared_val[10];
    {
      int j = (0) + threadIdx.x;
      for (int z = 0; z < 2; ++z)
        atomicAdd(&(shared_val[z]), j);
    }
  }
}

extern "C" __global__ __launch_bounds__(100) void _occa_buggy_kernel_1() {
  {
    int i = (0) + blockIdx.x;
    __shared__ float shared_val[10];
    {
      int j = (0) + threadIdx.x;
      if (j < 100)
        atomicAdd(&(shared_val[j]), j);
    }
  }
}
