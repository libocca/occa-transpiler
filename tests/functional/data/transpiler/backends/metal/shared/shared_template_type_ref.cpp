#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

template <class T>
struct ComplexType {
    T real;
    T imaginary;
};

kernel void _occa_function1_0(device const int* data [[buffer(0)]],
                              uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        threadgroup ComplexType<int> arr1[32];
        threadgroup ComplexType<float> arr2[8][32];
        { int j = (0) + _occa_thread_position.x; }
    }
}
