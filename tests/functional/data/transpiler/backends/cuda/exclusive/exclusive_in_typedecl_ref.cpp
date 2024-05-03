#include <cuda_runtime.h>
typedef float ex_float32_t;

extern "C" __global__ __launch_bounds__(32) void _occa_test_kernel_0() {
  {
    int i = (0) + blockIdx.x;
    ex_float32_t d[32];
    {
      int j = (0) + threadIdx.x;
      d[j] = i - j;
    }
  }
}
