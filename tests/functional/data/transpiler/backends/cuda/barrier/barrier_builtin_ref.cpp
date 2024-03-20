#include <cuda_runtime.h>

__device__ static float add(const float *a, int i, const float *b, int j) {
  return a[i] + b[j];
}

// TODO: fix preprocessor handling and try with define
extern "C" __global__
__launch_bounds__(4) void _occa_addVectors_0(const int N, const float *a,
                                             const float *b, float *ab) {
  {
    int i = (0) + ((4) * blockIdx.x);
    __shared__ float s_b[4];
    const float *g_a = a;
    {
      int j = (0) + threadIdx.x;
      s_b[j] = b[i + j];
      __syncthreads();
      ab[i + j] = add(g_a, i + j, s_b, j);
    }
  }
}
