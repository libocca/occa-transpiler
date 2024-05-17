#include <metal_compute>
#include <metal_stdlib>
using namespace metal;

const int offset = 1;

// template<typename T>
float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner
kernel void _occa_addVectors0_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i + ((1) * _occa_thread_position.x);
            if (i < entries) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner non 1 increment
kernel void _occa_addVectors1_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (0) + (((4) * 2) * _occa_group_position.x);
        {
            int i = _occa_tiled_i + ((2) * _occa_thread_position.x);
            if (i < entries) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner unary post add
kernel void _occa_addVectors2_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (0) + ((4) * _occa_group_position.x);
        {
            int i = _occa_tiled_i + _occa_thread_position.x;
            if (i < entries) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner unary pre add
kernel void _occa_addVectors3_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (0) + ((4) * _occa_group_position.x);
        {
            int i = _occa_tiled_i + _occa_thread_position.x;
            if (i < entries) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner, check=True
kernel void _occa_addVectors4_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i + ((1) * _occa_thread_position.x);
            if (i < entries) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner, complex range
kernel void _occa_addVectors5_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i =
            ((entries - 12 + 4)) + (((4) * (entries / 16 + 1)) * _occa_group_position.x);
        {
            int i = _occa_tiled_i + (((entries / 16 + 1)) * _occa_thread_position.x);
            if (i < (entries + 16)) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner, set dimension
kernel void _occa_addVectors6_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * _occa_group_position.y);
        {
            int i = _occa_tiled_i + ((1) * _occa_thread_position.z);
            if (i < entries) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}

// Outer -> inner ==> inner -> inner (nested)
kernel void _occa_addVectors7_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i + ((1) * _occa_thread_position.x);
            if (i < entries) {
                {
                    int _occa_tiled_j = (0) + (((4) * 1) * _occa_thread_position.y);
                    {
                        int j = _occa_tiled_j + ((1) * _occa_thread_position.z);
                        if (j < entries) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> inner ==> inner -> inner (nested) + complex range + check true
kernel void _occa_addVectors8_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = ((entries - 12 + static_cast<int>(*a))) +
                            (((4) * (entries / 16 + 1)) * _occa_group_position.x);
        {
            int i = _occa_tiled_i + (((entries / 16 + 1)) * _occa_thread_position.x);
            if (i < (entries + 16)) {
                {
                    unsigned long long _occa_tiled_j =
                        ((entries - 12 + static_cast<int>(*a))) +
                        (((4) * (entries / 16 + 1)) * _occa_thread_position.y);
                    {
                        unsigned long long j =
                            _occa_tiled_j + (((entries / 16 + 1)) * _occa_thread_position.z);
                        if (j < (entries + 16)) {
                            ab[i] = add(a[i], b[j]);
                        }
                    }
                }
            }
        }
    }
}

// Outer -> inner, <=
kernel void _occa_addVectors9_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int _occa_tiled_i = (0) + (((4) * 1) * _occa_group_position.x);
        {
            int i = _occa_tiled_i + ((1) * _occa_thread_position.x);
            if (i <= entries) {
                ab[i] = add(a[i], b[i]);
            }
        }
    }
}
