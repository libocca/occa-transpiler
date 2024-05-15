#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

namespace A {
template <class T>
struct Complex {
    T real;
    T imaginary;
};

namespace B {
struct Configs {
    unsigned int size1;
    unsigned long size2;
};

namespace C {
typedef int SIZE_TYPE;
typedef SIZE_TYPE SIZES;
}  // namespace C
}  // namespace B
}  // namespace A

kernel void _occa_function1_0(device const A::Complex<float>* __restrict__ vectorData [[buffer(0)]],
                              constant unsigned int& vectorSize [[buffer(1)]],
                              device const A::Complex<float>** __restrict__ matricesData
                              [[buffer(2)]],
                              device const A::B::Configs* __restrict__ matricesSizes [[buffer(3)]],
                              uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}

kernel void _occa_function2_0(device const A::Complex<float>* __restrict__ vectorData [[buffer(0)]],
                              device const A::B::Configs* __restrict__ configs [[buffer(1)]],
                              device A::B::C::SIZES* __restrict__ vectorSize [[buffer(2)]],
                              uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}
