#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

// const array
const int arr_const0[12] = {0};
int const arr_const1[12] = {0};
// Stupid formatting
const int arr_const2[12] = {0};
// Deduced size
const float arr_const3[] = {1., 2., 3., 4., 5., 6.};
// Multidimensional
const float arr_const4[][2] = {{1., 2.}, {3., 4.}, {5., 6.}};
const float arr_const5[][3][2] = {{{1., 2.}, {3., 4.}, {5., 6.}}, {{1., 2.}, {3., 4.}, {5., 6.}}};

// At least one @kern function is required
kernel void _occa_kern_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                         uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}
