#include <hip/hip_runtime.h>
__device__ static float add1(const float* a, int i, const float* b, int j) {
    return a[i] + b[i];
}

__device__ float add2(const float* a, int i, const float* b, int j) {
    return a[i] + b[i];
}
