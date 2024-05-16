#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_atomic_and_builtin_0(device const unsigned int* masks [[buffer(0)]],
                                       device unsigned int* mask [[buffer(1)]],
                                       uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                       uint3 _occa_thread_position
                                       [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            *mask ^= masks[0];
        }
    }
}

struct ComplexMaskType {
    unsigned int mask1;
    unsigned int mask2;
};

kernel void _occa_atomic_and_struct_0(device const ComplexMaskType* masks [[buffer(0)]],
                                      device ComplexMaskType* mask [[buffer(1)]],
                                      uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                      uint3 _occa_thread_position
                                      [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            mask->mask1 ^= masks[0].mask1;
            mask->mask2 ^= masks[0].mask2;
        }
    }
}
