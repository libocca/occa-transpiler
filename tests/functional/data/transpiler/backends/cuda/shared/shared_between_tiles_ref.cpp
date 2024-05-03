#include <cuda_runtime.h>

extern "C" __global__ __launch_bounds__(12) void _occa_test_kern_0() {
  {
    int _occa_tiled_i = (0) + ((4) * blockIdx.x);
    for (int i = _occa_tiled_i; i < (_occa_tiled_i + (4)); ++i) {
      if (i < 10) {
        __shared__ int shm[10];
        {
          int _occa_tiled_j = (0) + ((4) * threadIdx.y);
          {
            int j = _occa_tiled_j + threadIdx.x;
            if (j < 10) {
              shm[j] = j;
            }
          }
        }
        __syncthreads();
      }
    }
  }
}
