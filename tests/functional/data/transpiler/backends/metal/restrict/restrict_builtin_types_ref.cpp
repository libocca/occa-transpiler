#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_function1_0(device const int* __restrict__ i32Data [[buffer(0)]],
                              device float* __restrict__ fp32Data [[buffer(1)]],
                              device const double* __restrict__ fp64Data [[buffer(2)]],
                              uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            float* b = &fp32Data[0];
        }
    }
}
