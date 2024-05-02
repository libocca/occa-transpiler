#include <cuda_runtime.h>

struct Data {
  float *__restrict__ data;
  int *__restrict__ idxs;
};

__constant__ int SIZE = 128;
__device__ float add(float a, float b);

__device__ float add2(float a, float b) { return a + b; }

extern "C" __global__
__launch_bounds__(1) void _occa_function1_0(const Data data1,
                                            const Data data2) {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
