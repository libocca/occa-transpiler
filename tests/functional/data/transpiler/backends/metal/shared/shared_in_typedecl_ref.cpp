#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

typedef float sh_float32_t;

kernel void _occa_test_kernel_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        threadgroup sh_float32_t b[32];
        {
            int j = (0) + _occa_thread_position.x;
            b[j] = i + j;
        }
    }
}
