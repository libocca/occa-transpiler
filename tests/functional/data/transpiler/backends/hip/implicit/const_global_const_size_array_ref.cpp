#include <cuda_runtime.h>
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
