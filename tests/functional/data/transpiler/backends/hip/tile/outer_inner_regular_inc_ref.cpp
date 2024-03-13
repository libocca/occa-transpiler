#include <hip/hip_runtime.h>
__constant__ int offset = 1;

// template<typename T>
__device__ float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner ==> regular -> regular
extern "C" __global__ void _occa_addVectors0_0(const int entries,
                                               const float* a,
                                               const float* b,
                                               float* ab) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * blockIdx.x);
        {
            int i = _occa_tiled_i + ((1) * threadIdx.x);
            if (i < entries) {
                for (int _occa_tiled_j = 0; _occa_tiled_j < entries; _occa_tiled_j += (4)) {
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + (4)); ++j)
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                }
            }
        }
    }
}

// Outer -> inner ==> inner -> regular
extern "C" __global__ void _occa_addVectors2_0(const int entries,
                                               const float* a,
                                               const float* b,
                                               float* ab) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * blockIdx.x);
        {
            int i = _occa_tiled_i + ((1) * threadIdx.x);
            if (i < entries) {
                {
                    int _occa_tiled_j = (0) + ((4) * threadIdx.y);
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + (4)); ++j)
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                }
            }
        }
    }
}

// Outer -> inner ==> inner -> inner
extern "C" __global__ void _occa_addVectors3_0(const int entries,
                                               const float* a,
                                               const float* b,
                                               float* ab) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * blockIdx.x);
        {
            int i = _occa_tiled_i + ((1) * threadIdx.x);
            if (i < entries) {
                {
                    int _occa_tiled_j = (0) + ((4) * threadIdx.y);
                    {
                        int j = _occa_tiled_j + threadIdx.y;
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> outer ==> inner -> regular
extern "C" __global__ void _occa_addVectors4_0(const int entries,
                                               const float* a,
                                               const float* b,
                                               float* ab) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * blockIdx.x);
        {
            int i = _occa_tiled_i + ((1) * blockIdx.y);
            if (i < entries) {
                {
                    int _occa_tiled_j = (0) + ((4) * threadIdx.y);
                    for (int j = _occa_tiled_j; j < (_occa_tiled_j + (4)); ++j)
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                }
            }
        }
    }
}

// Outer -> outer ==> inner -> inner
extern "C" __global__ void _occa_addVectors5_0(const int entries,
                                               const float* a,
                                               const float* b,
                                               float* ab) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * blockIdx.x);
        {
            int i = _occa_tiled_i + ((1) * blockIdx.y);
            if (i < entries) {
                {
                    int _occa_tiled_j = (0) + ((4) * threadIdx.y);
                    {
                        int j = _occa_tiled_j + threadIdx.z;
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> outer ==> outer -> inner
extern "C" __global__ void _occa_addVectors6_0(const int entries,
                                               const float* a,
                                               const float* b,
                                               float* ab) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * blockIdx.x);
        {
            int i = _occa_tiled_i + ((1) * blockIdx.y);
            if (i < entries) {
                {
                    int _occa_tiled_j = (0) + ((4) * blockIdx.z);
                    {
                        int j = _occa_tiled_j + threadIdx.x;
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}
