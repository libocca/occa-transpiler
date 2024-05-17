#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_test_kernel_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.y;
        {
            int i2 = (0) + _occa_group_position.x;
            { int j = (0) + _occa_thread_position.x; }
            for (int ii = 0; ii < 10; ++ii) {
                {
                    int j = (0) + _occa_thread_position.x;
                }
                for (int j = 0; j < 10; ++j) {
                }
            }
        }
        for (int ii = 0; ii < 10; ++ii) {
            {
                int i = (0) + _occa_group_position.x;
                { int j = (0) + _occa_thread_position.x; }
            }
        }
    }
}

kernel void _occa_test_kernel_1(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.y;
        for (int i2 = 0; i2 < 10; ++i2) {
            {
                int i2 = (0) + _occa_group_position.x;
                { int j = (0) + _occa_thread_position.x; }
            }
        }
    }
}
