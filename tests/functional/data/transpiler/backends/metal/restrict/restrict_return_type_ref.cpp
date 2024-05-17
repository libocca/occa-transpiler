#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

float* __restrict__ myfn(float* a) {
    return a + 1;
}

float* myfn2(float* a) {
    return a + 1;
}

kernel void _occa_hello_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                          uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}
