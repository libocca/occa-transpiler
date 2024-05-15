#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

struct S {
    int hello[12];
};

extern const int arr_0[];
extern const float arr_1[];
extern const S arr_2[];

// At least one @kern function is required
kernel void _occa_kern_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                         uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        { int j = (0) + _occa_thread_position.x; }
    }
}
