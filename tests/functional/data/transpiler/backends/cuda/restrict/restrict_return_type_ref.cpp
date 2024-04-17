#include <cuda_runtime.h>

float *__restrict__ myfn(float *a) { return a + 1; }

extern "C" __global__ __launch_bounds__(10) void _occa_hello_0() {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
