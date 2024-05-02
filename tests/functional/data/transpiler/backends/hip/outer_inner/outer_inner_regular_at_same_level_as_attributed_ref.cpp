#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(10) void _occa_test_kernel_0() {
  {
    int i = (0) + blockIdx.y;
    {
      int i2 = (0) + blockIdx.x;
      { int j = (0) + threadIdx.x; }
      for (int ii = 0; ii < 10; ++ii) {
        {
          int j = (0) + threadIdx.x;
        }
        for (int j = 0; j < 10; ++j) {
        }
      }
    }
    for (int ii = 0; ii < 10; ++ii) {
      {
        int i = (0) + blockIdx.x;
        { int j = (0) + threadIdx.x; }
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(10) void _occa_test_kernel_1() {
  {
    int i = (0) + blockIdx.y;
    for (int i2 = 0; i2 < 10; ++i2) {
      {
        int i2 = (0) + blockIdx.x;
        { int j = (0) + threadIdx.x; }
      }
    }
  }
}
