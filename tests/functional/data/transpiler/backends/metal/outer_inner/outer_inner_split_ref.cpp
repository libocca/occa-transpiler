#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

kernel void _occa_test0_0(constant int& entries [[buffer(0)]],
                          device const float* a [[buffer(1)]],
                          device const float* b [[buffer(2)]],
                          device float* ab [[buffer(3)]],
                          uint3 _occa_group_position [[threadgroup_position_in_grid]],
                          uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int x = (0) + _occa_group_position.z;
        // int before1 = 1 + before0;
        int before1 = 1;
        {
            int y = (0) + _occa_group_position.y;
            int before2 = 1 + before1;
            {
                int z = (0) + _occa_group_position.x;
                int before3 = 1 + before2;
                {
                    int n = (0) + _occa_thread_position.z;
                    int after0 = 1 + before3;
                    {
                        int m = (0) + _occa_thread_position.y;
                        int after1 = 1 + after0;
                        {
                            int k = (0) + _occa_thread_position.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        {
                            int k = (0) + _occa_thread_position.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                    {
                        int m = (0) + _occa_thread_position.y;
                        int after1 = 1 + after0;
                        {
                            int k = (0) + _occa_thread_position.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        {
                            int k = (0) + _occa_thread_position.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                }
            }
        }
    }
}

kernel void _occa_test0_1(constant int& entries [[buffer(0)]],
                          device const float* a [[buffer(1)]],
                          device const float* b [[buffer(2)]],
                          device float* ab [[buffer(3)]],
                          uint3 _occa_group_position [[threadgroup_position_in_grid]],
                          uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int x = (0) + _occa_group_position.z;
        // int before1 = 1 + before00;
        int before1 = 1;
        {
            int y = (0) + _occa_group_position.y;
            int before2 = 1 + before1;
            {
                int z = (0) + _occa_group_position.x;
                int before3 = 1 + before2;
                {
                    int n = (0) + _occa_thread_position.z;
                    int after0 = 1 + before3;
                    {
                        int m = (0) + _occa_thread_position.y;
                        int after1 = 1 + after0;
                        {
                            int k = (0) + _occa_thread_position.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        {
                            int k = (0) + _occa_thread_position.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                    {
                        int m = (0) + _occa_thread_position.y;
                        int after1 = 1 + after0;
                        {
                            int k = (0) + _occa_thread_position.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        {
                            int k = (0) + _occa_thread_position.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                }
            }
        }
    }
}
