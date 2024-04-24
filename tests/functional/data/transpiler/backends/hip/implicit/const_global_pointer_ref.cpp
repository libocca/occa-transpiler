#include <hip/hip_runtime.h>
// pointer to const
__constant__ int *ptr_const0 = 0;
__constant__ int *ptr_const1 = 0;
// const pointer to const
__constant__ int *const ptr_const2 = 0;
__constant__ int *const ptr_const3 = 0;
// const pointer to non const
int *const ptr_const4 = 0;
// Stupid formatting
__constant__ int *ptr_const5 = 0;

// At least one @kern function is requried
extern "C" __global__ __launch_bounds__(32) void _occa_kern_0() {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
