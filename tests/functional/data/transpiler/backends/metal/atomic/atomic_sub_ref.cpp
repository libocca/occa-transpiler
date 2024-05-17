#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_atomic_sub_builtin_0(device const int* iVec [[buffer(0)]],
                                       device int* iSum [[buffer(1)]],
                                       device const float* fVec [[buffer(2)]],
                                       device float* fSum [[buffer(3)]],
                                       uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                       uint3 _occa_thread_position
                                       [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            *iSum -= iVec[0];
            *fSum -= fVec[0];
        }
    }
}

struct ComplexTypeF32 {
    float real;
    float imag;
};

kernel void _occa_atomic_sub_struct_0(device const ComplexTypeF32* vec [[buffer(0)]],
                                      device ComplexTypeF32* sum [[buffer(1)]],
                                      uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                      uint3 _occa_thread_position
                                      [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            sum->real -= vec[0].real;
            sum->imag -= vec[0].imag;
        }
    }
}

template <class T>
struct ComplexType {
    T real;
    T imag;
};

kernel void _occa_atomic_sub_template_0(device const ComplexType<float>* vec [[buffer(0)]],
                                        device ComplexType<float>* sum [[buffer(1)]],
                                        uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                        uint3 _occa_thread_position
                                        [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        {
            int j = (0) + _occa_thread_position.x;
            sum->real -= vec[0].real;
            sum->imag -= vec[0].imag;
        }
    }
}
