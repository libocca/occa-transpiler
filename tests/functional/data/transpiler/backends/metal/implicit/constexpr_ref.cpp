#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

constexpr float f = 13;

class HelloClass {
   public:
    static constexpr int a = 2 + 2;
};

kernel void _occa_test_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                         uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}
