#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_test_kern_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
