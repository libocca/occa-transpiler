#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

template <class T>
struct Complex {
    T real;
    T imaginary;
};

struct Configs {
    unsigned int size1;
    unsigned long size2;
};

struct Data {
    float* __restrict__ x;
    float* __restrict__ y;
    unsigned long size;
};

kernel void _occa_function1_0(device const Complex<float>* __restrict__ vectorData [[buffer(0)]],
                              constant unsigned int& vectorSize [[buffer(1)]],
                              device const Complex<float>** __restrict__ matricesData [[buffer(2)]],
                              device const Configs* __restrict__ matricesSizes [[buffer(3)]],
                              uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}
