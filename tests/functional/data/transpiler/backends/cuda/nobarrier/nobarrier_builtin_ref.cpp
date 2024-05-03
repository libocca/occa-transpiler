#include <cuda_runtime.h>

extern "C" __global__ __launch_bounds__(10) void _occa_hello_kern_0() {
  {
    int i = (0) + blockIdx.x;
    __shared__ int shm[10];
    {
      int j = (0) + threadIdx.x;
      shm[j] = j;
    }
    __syncthreads();
    {
      int j = (0) + threadIdx.x;
      shm[j] = j;
    }
    {
      int j = (0) + threadIdx.x;
      shm[j] = j;
    }
    __syncthreads();
    {
      int j = (0) + threadIdx.x;
      shm[j] = j;
    }
  }
}

extern "C" __global__ __launch_bounds__(32) void _occa_priority_issue_0() {
  {
    int i = (0) + blockIdx.x;
    __shared__ float shm[32];
    {
      int j = (0) + threadIdx.x;
      shm[i] = i;
    }
    {
      int j = (0) + threadIdx.x;
      atomicAdd(&(shm[i * j]), 32);
    }
  }
}
