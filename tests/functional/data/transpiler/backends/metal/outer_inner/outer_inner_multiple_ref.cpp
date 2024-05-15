#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// with shared memory usage (should be automatic sync)
kernel void _occa_addVectors_0(constant int& entries [[buffer(0)]],
                               device float* a [[buffer(1)]],
                               device float* b [[buffer(2)]],
                               device float* ab [[buffer(3)]],
                               device float* mat [[buffer(4)]],
                               uint3 _occa_group_position [[threadgroup_position_in_grid]],
                               uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + ((1) * _occa_group_position.y);
        {
            int i2 = (0) + ((1) * _occa_group_position.x);
            threadgroup int shm[32];
            threadgroup int shm2[32];
            {
                int j = (0) + ((1) * _occa_thread_position.z);
                shm[j] = 0;  // shared memory usage -> should be barrier after @inner loop
                mat[0 + (10 * (0))] = 12;
                {
                    int k = (0) + ((1) * _occa_thread_position.y);
                    {
                        int ii = (0) + ((1) * _occa_thread_position.x);
                        ab[i] = add(a[i], b[k]);
                    }
                    ab[i] = add(a[i], b[k]);
                }
                {
                    int k = (0) + ((1) * _occa_thread_position.y);
                    {
                        int ii = (0) + ((1) * _occa_thread_position.x);
                        ab[i] = add(a[i], b[k]);
                    }
                    ab[i] = add(a[i], b[k]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                int _occa_tiled_j = (0) + (((4) * 1) * _occa_thread_position.z);
                {
                    int j = _occa_tiled_j + ((1) * _occa_thread_position.y);
                    {
                        {
                            int k = (0) + ((1) * _occa_thread_position.x);
                            // shared memory usage -> should be barrier, since @tile is inner,
                            // inner
                            shm[j] = 0;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                int j = (0) + ((1) * _occa_thread_position.z);
                shm[j] = 0;
                {
                    int k = (0) + ((1) * _occa_thread_position.y);
                    {
                        int ii = (0) + ((1) * _occa_thread_position.x);
                        ab[i] = add(a[i], b[k]);
                    }
                    ab[i] = add(a[i], b[k]);
                }
                {
                    int _occa_tiled_k = (0) + (((4) * 1) * _occa_thread_position.y);
                    {
                        int k = _occa_tiled_k + ((1) * _occa_thread_position.x);
                        { ab[i] = add(a[i], b[k]); }
                    }
                }
            }
        }
    }
}

// without shared memory usage (should be no automatic sync)
kernel void _occa_addVectors1_0(constant int& entries [[buffer(0)]],
                                device float* a [[buffer(1)]],
                                device float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                device float* mat [[buffer(4)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + ((1) * _occa_group_position.y);
        {
            int i2 = (0) + ((1) * _occa_group_position.x);
            threadgroup int shm[32];
            threadgroup int shm2[32];
            {
                int j = (0) + ((1) * _occa_thread_position.z);
                // shm[j] = 0; // shared memory usage -> should be barrier after @inner
                // loop
                mat[0 + (10 * (0))] = 12;
                {
                    int k = (0) + ((1) * _occa_thread_position.y);
                    {
                        int ii = (0) + ((1) * _occa_thread_position.x);
                        ab[i] = add(a[i], b[k]);
                    }
                    ab[i] = add(a[i], b[k]);
                }
                {
                    int k = (0) + ((1) * _occa_thread_position.y);
                    {
                        int ii = (0) + ((1) * _occa_thread_position.x);
                        ab[i] = add(a[i], b[k]);
                    }
                    ab[i] = add(a[i], b[k]);
                }
            }
            {
                int _occa_tiled_j = (0) + (((4) * 1) * _occa_thread_position.z);
                {
                    int j = _occa_tiled_j + ((1) * _occa_thread_position.y);
                    {
                        {
                            int k = (0) + ((1) * _occa_thread_position.x);
                            // shared memory usage -> should be barrier, since @tile is inner,
                            // inner shm[j] = 0;
                        }
                    }
                }
            }
            {
                int j = (0) + ((1) * _occa_thread_position.z);
                shm[j] = 0;
                {
                    int k = (0) + ((1) * _occa_thread_position.y);
                    {
                        int ii = (0) + ((1) * _occa_thread_position.x);
                        ab[i] = add(a[i], b[k]);
                    }
                    ab[i] = add(a[i], b[k]);
                }
                {
                    int _occa_tiled_k = (0) + (((4) * 1) * _occa_thread_position.y);
                    {
                        int k = _occa_tiled_k + ((1) * _occa_thread_position.x);
                        { ab[i] = add(a[i], b[k]); }
                    }
                }
            }
        }
    }
}
