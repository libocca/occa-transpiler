#include <cuda_runtime.h>

extern "C" __global__ __launch_bounds__(30) void _occa_test0_0(const int entries,
                                                               const float* a,
                                                               const float* b,
                                                               float* ab) {
    {
        int x = (0) + blockIdx.z;
        // int before1 = 1 + before0;
        int before1 = 1;
        {
            int y = (0) + blockIdx.y;
            int before2 = 1 + before1;
            {
                int z = (0) + blockIdx.x;
                int before3 = 1 + before2;
                {
                    int n = (0) + threadIdx.z;
                    int after0 = 1 + before3;
                    {
                        int m = (0) + threadIdx.y;
                        int after1 = 1 + after0;
                        {
                            int k = (0) + threadIdx.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        {
                            int k = (0) + threadIdx.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                    {
                        int m = (0) + threadIdx.y;
                        int after1 = 1 + after0;
                        {
                            int k = (0) + threadIdx.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        {
                            int k = (0) + threadIdx.x;
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

extern "C" __global__ __launch_bounds__(105) void _occa_test0_1(const int entries,
                                                                const float* a,
                                                                const float* b,
                                                                float* ab) {
    {
        int x = (0) + blockIdx.z;
        // int before1 = 1 + before00;
        int before1 = 1;
        {
            int y = (0) + blockIdx.y;
            int before2 = 1 + before1;
            {
                int z = (0) + blockIdx.x;
                int before3 = 1 + before2;
                {
                    int n = (0) + threadIdx.z;
                    int after0 = 1 + before3;
                    {
                        int m = (0) + threadIdx.y;
                        int after1 = 1 + after0;
                        {
                            int k = (0) + threadIdx.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        {
                            int k = (0) + threadIdx.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                    }
                    {
                        int m = (0) + threadIdx.y;
                        int after1 = 1 + after0;
                        {
                            int k = (0) + threadIdx.x;
                            int after2 = 1 + after1;
                            ab[x] =
                                a[x] + b[x] + static_cast<float>(k + m + n + z + y + x + after2);
                        }
                        {
                            int k = (0) + threadIdx.x;
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
