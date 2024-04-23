#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(10) void _occa_test_kern_0() {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      __syncwarp();
    }
  }
}
