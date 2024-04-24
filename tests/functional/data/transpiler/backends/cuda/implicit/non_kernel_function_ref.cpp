#include <cuda_runtime.h>

__device__ static float add1(const float *a, int i, const float *b, int j) {
  return a[i] + b[i];
}

__device__ float add2(const float *a, int i, const float *b, int j) {
  return a[i] + b[i];
}

// At least one @kern function is requried
extern "C" __global__ __launch_bounds__(32) void _occa_kern_0() {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
