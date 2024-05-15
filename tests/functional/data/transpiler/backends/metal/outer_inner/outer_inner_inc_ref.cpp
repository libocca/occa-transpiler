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
        int j = (0) + ((1) * _occa_group_position.x);
        {
            int i = (0) + ((1) * _occa_thread_position.x);
            ab[i] = add(a[i], b[i]);
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
        int j = (0) + ((2) * _occa_group_position.x);
        {
            int i = (0) + ((2) * _occa_thread_position.x);
            ab[i] = add(a[i], b[i]);
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
        int j = (0) + _occa_group_position.x;
        {
            int i = (0) + _occa_thread_position.x;
            ab[i] = add(a[i], b[i]);
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
        int j = (0) + _occa_group_position.x;
        {
            int i = (0) + _occa_thread_position.x;
            ab[i] = add(a[i], b[i]);
        }
    }
}

// Outer -> outer -> inner -> inner
// TODO: change after sema calculates dimensions
kernel void _occa_addVectors4_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.y;
        {
            int j = (0) + _occa_group_position.x;
            {
                int k = (0) + _occa_thread_position.y;
                {
                    int ii = (0) + _occa_thread_position.x;
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + manual dimensions specification
kernel void _occa_addVectors5_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.y;
        {
            int j = (0) + _occa_group_position.x;
            {
                int k = (0) + _occa_thread_position.y;
                {
                    int ii = (0) + _occa_thread_position.x;
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}

// Outer -> outer -> inner -> inner + partially manual dimensions specification
kernel void _occa_addVectors6_0(constant int& entries [[buffer(0)]],
                                device const float* a [[buffer(1)]],
                                device const float* b [[buffer(2)]],
                                device float* ab [[buffer(3)]],
                                uint3 _occa_group_position [[threadgroup_position_in_grid]],
                                uint3 _occa_thread_position [[thread_position_in_threadgroup]]) {
    {
        int i = (0) + _occa_group_position.y;
        {
            int j = (0) + _occa_group_position.x;
            {
                int k = (0) + _occa_thread_position.y;
                {
                    int ii = (0) + _occa_thread_position.x;
                    ab[ii + k] = add(a[i], b[j]);
                }
            }
        }
    }
}
