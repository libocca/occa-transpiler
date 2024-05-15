#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

// int const, const int
const int var_const0 = 0;
int const var_const1 = 0;
// volatile qualifier
volatile const int var_const2 = 0;
volatile int const var_const3 = 0;
// Stupid formatting
const int var_const4 = 0;
int const var_const5 = 0;

// At least one @kern function is required
kernel void _occa_kern_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                         uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}
