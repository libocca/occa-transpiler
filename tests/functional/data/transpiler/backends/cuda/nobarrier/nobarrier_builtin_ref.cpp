#include <cuda_runtime.h>

extern "C" __global__ __launch_bounds__(10) void _occa_hello_kern_0() {
    {
        int i = (0) + blockIdx.x;
        __shared__ int shm[10];
        {
            int j = (0) + threadIdx.x;
            shm[j] = j;
            __syncthreads();
        }
        {
            int j = (0) + threadIdx.x;
            shm[j] = j;
        }
        {
            int j = (0) + threadIdx.x;
            shm[j] = j;
            __syncthreads();
        }
        {
            int j = (0) + threadIdx.x;
            shm[j] = j;
        }
    }
}
