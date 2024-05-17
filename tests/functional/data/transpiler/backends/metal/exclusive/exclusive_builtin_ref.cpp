#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

static float add(const float* a, int i, const float* b, int j) {
    return a[i] + b[j];
}

// TODO: fix preprocessor handling and try with define
// #define BLOCK_SIZE 4
const int BLOCK_SIZE = 4;

kernel void _occa_addVectors_0(constant int& N [[buffer(0)]],
                               device const float* a [[buffer(1)]],
                               device const float* b [[buffer(2)]],
                               device float* ab [[buffer(3)]],
                               uint3 _occa_group_position [[threadgroup_position_in_grid]],
                               uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + ((BLOCK_SIZE)*_occa_group_position.x);
        threadgroup float s_b[BLOCK_SIZE];
        const float* g_a = a;
        {
            int j = (0) + _occa_thread_position.x;
            s_b[j] = b[i + j];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ab[i + j] = add(g_a, i + j, s_b, j);
        }
    }
}
