#include <cuda_runtime.h>

struct S {
  int hello[12];
};

extern __constant__ int arr_0[];
extern __constant__ float arr_1[];
extern __constant__ S arr_2[];

// At least one @kern function is requried
extern "C" __global__ __launch_bounds__(32) void _occa_kern_0() {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
