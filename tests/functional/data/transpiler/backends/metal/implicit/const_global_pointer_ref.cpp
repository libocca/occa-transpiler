#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

// pointer to const
const int* ptr_const0 = 0;
int const* ptr_const1 = 0;
// const pointer to const
const int* const ptr_const2 = 0;
int const* const ptr_const3 = 0;
// const pointer to non const
int* const ptr_const4 = 0;
// Stupid formatting
const int* ptr_const5 = 0;

// At least one @kern function is required
kernel void _occa_kern_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                         uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}
