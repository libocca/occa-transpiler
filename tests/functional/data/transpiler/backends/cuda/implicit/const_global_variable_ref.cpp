#include <cuda_runtime.h>
// int const, const int
__constant__ int var_const0 = 0;
__constant__ int var_const1 = 0;
// volatile qualifier
volatile __constant__ int var_const2 = 0;
volatile __constant__ int var_const3 = 0;
// Stupid formatting
__constant__ int var_const4 = 0;
__constant__ int var_const5 = 0;

// At least one @kern function is requried
extern "C" __global__ __launch_bounds__(32) void _occa_kern_0() {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
