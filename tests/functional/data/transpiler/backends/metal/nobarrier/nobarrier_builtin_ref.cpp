#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_hello_kern_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                               uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        threadgroup int shm[10];
        {
            int j = (0) + _occa_thread_position.x;
            shm[j] = j;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            int j = (0) + _occa_thread_position.x;
            shm[j] = j;
        }
        {
            int j = (0) + _occa_thread_position.x;
            shm[j] = j;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            int j = (0) + _occa_thread_position.x;
            shm[j] = j;
        }
    }
}

kernel void _occa_priority_issue_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                   uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        threadgroup float shm[32];
        {
            int j = (0) + _occa_thread_position.x;
            shm[i] = i;
        }
        {
            int j = (0) + _occa_thread_position.x;
            [[okl_atomic("")]] shm[i * j] += 32;
        }
    }
}
