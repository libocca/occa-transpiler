#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_function1_0(device const int* data [[buffer(0)]],
                              uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        threadgroup int arr1[32];
        threadgroup float arr2[8][32];
        threadgroup double arr3[4 + 4];
        { int j = (0) + _occa_thread_position.x; }
    }
}

// syncronization between @inner loops:
kernel void _occa_function2_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        threadgroup int shm[10];
        {
            int j = (0) + _occa_thread_position.x;
            shm[i] = j;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // sync should be here
        {
            int j = (0) + _occa_thread_position.x;
            shm[i] = j;
        }
        // sync should not be here
    }
}

// Even if loop is last, if it is inside regular loop, syncronization is
// inserted
kernel void _occa_function3_0(uint3 _occa_group_position [[threadgroup_position_in_grid]],
                              uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.x;
        threadgroup int shm[10];
        for (int q = 0; q < 5; ++q) {
            {
                int j = (0) + _occa_thread_position.x;
                shm[i] = j;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // sync should be here
        }
    }
}
