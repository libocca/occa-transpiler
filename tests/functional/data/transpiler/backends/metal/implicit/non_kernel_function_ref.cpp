#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

static float add1(const float* a, int i, const float* b, int j) {
    return a[i] + b[i];
}

float add2(const float* a, int i, const float* b, int j) {
    return a[i] + b[i];
}

// At least one @kern function is required
kernel void _occa_kern_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                         uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}
