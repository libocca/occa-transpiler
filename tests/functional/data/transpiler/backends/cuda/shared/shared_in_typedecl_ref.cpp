#include <cuda_runtime.h>
typedef __shared__ float sh_float32_t;

extern "C" __global__ __launch_bounds__(32) void _occa_test_kernel_0() {
  {
    int i = (0) + blockIdx.x;
    __shared__ sh_float32_t b[32];
    {
      int j = (0) + threadIdx.x;
      b[j] = i + j;
    }
  }
}
