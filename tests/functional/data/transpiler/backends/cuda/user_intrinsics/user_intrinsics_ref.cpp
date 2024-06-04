#include <cuda_runtime.h>

// INFO: transpiling to cuda backend already include necessary header
__device__ bool okl_is_nan(float value) { return isnan(value) != 0; }

extern "C" __global__ __launch_bounds__(32) void _occa_zero_nans_0(float *vec) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      int idx = i * 32 + j;
      float value = vec[idx];
      if (okl_is_nan(value)) {
        vec[idx] = 0.0f;
      }
    }
  }
}

