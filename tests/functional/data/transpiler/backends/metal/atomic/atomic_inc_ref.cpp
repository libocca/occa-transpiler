#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_atomic_inc_builtin_0(device unsigned int* value [[buffer(0)]],
                                       uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                       uint3 _occa_thread_position
                                       [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            ++(*value);
        }
    }
}

struct ComplexMaskType {
    unsigned int val1;
    int val2;
};

kernel void _occa_atomic_inc_struct_0(device ComplexMaskType* value [[buffer(0)]],
                                      uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                      uint3 _occa_thread_position
                                      [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            ++value->val1;
            value->val2++;
        }
    }
}
