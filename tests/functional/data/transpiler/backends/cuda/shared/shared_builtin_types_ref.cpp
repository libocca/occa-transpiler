#include <cuda_runtime.h>

extern "C" __global__
__launch_bounds__(64) void _occa_function1_0(const int *data) {
  {
    int i = (0) + blockIdx.x;
    __shared__ int arr1[32];
    __shared__ float arr2[8][32];
    __shared__ double arr3[4 + 4];
    { int j = (0) + threadIdx.x; }
  }
}

// syncronization between @inner loops:
extern "C" __global__ __launch_bounds__(10) void _occa_function2_0() {
  {
    int i = (0) + blockIdx.x;
    __shared__ int shm[10];
    {
      int j = (0) + threadIdx.x;
      shm[i] = j;
    }
    __syncthreads();
    // sync should be here
    {
      int j = (0) + threadIdx.x;
      shm[i] = j;
    }
    // sync should not be here
  }
}

// Even if loop is last, if it is inside regular loop, syncronization is
// inserted
extern "C" __global__ __launch_bounds__(10) void _occa_function3_0() {
  {
    int i = (0) + blockIdx.x;
    __shared__ int shm[10];
    for (int q = 0; q < 5; ++q) {
      {
        int j = (0) + threadIdx.x;
        shm[i] = j;
      }
      __syncthreads();
      // sync should be here
    }
  }
}
