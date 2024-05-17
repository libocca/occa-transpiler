#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

struct ComplexTypeF32 {
    ComplexTypeF32& operator=(const ComplexTypeF32&) = default;
    float real;
    float imag;
};

kernel void _occa_atomic_exch_struct_0(device const ComplexTypeF32* vec [[buffer(0)]],
                                       device ComplexTypeF32* result [[buffer(1)]],
                                       uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                       uint3 _occa_thread_position
                                       [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            *result = vec[0];
        }
    }
}
