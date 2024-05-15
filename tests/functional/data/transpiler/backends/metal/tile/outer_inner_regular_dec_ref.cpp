#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

const int offset = 1;

float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner ==> regular -> regular
kernel void _occa_addVectors0_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (entries - 1) - (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i - ((1) * _occa_thread_position.x);
            if (i >= 0) {
                for (int _occa_tiled_j = entries; _occa_tiled_j > 0; _occa_tiled_j -= (4)) {
                    for (int j = _occa_tiled_j; j > (_occa_tiled_j - (4)); --j) {
                        if (j > 0) {
                            ab[i] = add(a[i], b[j - 1]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> inner ==> inner -> regular
kernel void _occa_addVectors2_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (entries - 1) - (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i - ((1) * _occa_thread_position.x);
            if (i >= 0) {
                {
                    int _occa_tiled_j = (entries) - ((4) * _occa_thread_position.y);
                    for (int j = _occa_tiled_j; j > (_occa_tiled_j - (4)); --j) {
                        if (j > 0) {
                            ab[i] = add(a[i], b[j - 1]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> inner ==> inner -> inner
kernel void _occa_addVectors3_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (entries - 1) - (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i - ((1) * _occa_thread_position.x);
            if (i >= 0) {
                {
                    int _occa_tiled_j = (entries) - ((4) * _occa_thread_position.y);
                    {
                        int j = _occa_tiled_j - _occa_thread_position.y;
                        if (j > 0) {
                            ab[i] = add(a[i], b[j - 1]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> outer ==> inner -> regular
kernel void _occa_addVectors4_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (entries - 1) - (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i - ((1) * _occa_group_position.y);
            if (i >= 0) {
                {
                    int _occa_tiled_j = (entries) - ((4) * _occa_thread_position.y);
                    for (int j = _occa_tiled_j; j > (_occa_tiled_j - (4)); --j) {
                        if (j > 0) {
                            ab[i] = add(a[i], b[j - 1]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> outer ==> inner -> inner
kernel void _occa_addVectors5_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (entries - 1) - (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i - ((1) * _occa_group_position.y);
            if (i >= 0) {
                {
                    int _occa_tiled_j = (entries) - ((4) * _occa_thread_position.y);
                    {
                        int j = _occa_tiled_j - _occa_thread_position.z;
                        if (j > 0) {
                            ab[i] = add(a[i], b[j - 1]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> outer ==> outer -> inner
kernel void _occa_addVectors6_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (entries - 1) - (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i - ((1) * _occa_group_position.y);
            if (i >= 0) {
                {
                    int _occa_tiled_j = (entries) - ((4) * _occa_group_position.z);
                    {
                        int j = _occa_tiled_j - _occa_thread_position.x;
                        if (j > 0) {
                            ab[i] = add(a[i], b[j - 1]);
                        }
                    }
                }
            }
        }
    }
}
