#include <cuda_runtime.h>


extern "C" __global__ __launch_bounds__(1) void _occa_function1_0(
    const int *__restrict__ i32Data, float *__restrict__ fp32Data,
    const double *__restrict__ fp64Data) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      float *__restrict__ b = &fp32Data[0];
    }
  }
}
