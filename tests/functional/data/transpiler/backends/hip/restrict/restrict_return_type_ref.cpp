#include <hip/hip_runtime.h>

__device__ float *__restrict__ myfn(float *a) { return a + 1; }

__device__ float *myfn2(float *a) { return a + 1; }

extern "C" __global__ __launch_bounds__(10) void _occa_hello_0() {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
