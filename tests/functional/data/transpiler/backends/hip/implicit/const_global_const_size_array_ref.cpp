#include <hip/hip_runtime.h>
// const array
__constant__ int arr_const0[12] = {0};
__constant__ int arr_const1[12] = {0};
// Stupid formatting
__constant__ int arr_const2[12] = {0};
// Deduced size
__constant__ float arr_const3[] = {1., 2., 3., 4., 5., 6.};
// Multidimensional
__constant__ float arr_const4[][2] = {{1., 2.}, {3., 4.}, {5., 6.}};
__constant__ float arr_const5[][3][2] = {{{1., 2.}, {3., 4.}, {5., 6.}},
                                         {{1., 2.}, {3., 4.}, {5., 6.}}};

// At least one @kern function is requried
extern "C" __global__ __launch_bounds__(32) void _occa_kern_0() {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
