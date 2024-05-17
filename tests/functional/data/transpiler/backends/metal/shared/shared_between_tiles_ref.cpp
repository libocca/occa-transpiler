#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_test_kern_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (0) + ((4) * _occa_group_position.x);
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + (4)); ++i) {
            if (i < 10) {
                threadgroup int shm[10];
                {
                    int _occa_tiled_j = (0) + ((4) * _occa_thread_position.y);
                    {
                        int j = _occa_tiled_j + _occa_thread_position.x;
                        if (j < 10) {
                            shm[j] = j;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
