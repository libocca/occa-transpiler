#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

namespace {
// Math functions

// Single precision
[[maybe_unused]] inline __device__ float okl_exp10f(float x) { return exp10f(x); }

// Warp Shuffle Functions
template <class T>
inline __device__ T okl_shfl_sync(unsigned mask, T var, int srcLane,
                                  int width = warpSize) {
  return __shfl_sync(mask, var, srcLane, width);
}

template <class T>
inline __device__ T okl_shfl_up_sync(unsigned mask, T var, unsigned int delta,
                                     int width = warpSize) {
  return __shfl_up_sync(mask, var, delta, width);
}

template <class T>
inline __device__ T okl_shfl_down_sync(unsigned mask, T var, unsigned int delta,
                                       int width = warpSize) {
  return __shfl_down_sync(mask, var, delta, width);
}

template <class T>
inline __device__ T okl_shfl_xor_sync(unsigned mask, T var, int laneMask,
                                      int width = warpSize) {
  return __shfl_xor_sync(mask, var, laneMask, width);
}

// Pipeline Primitives Interface
[[maybe_unused]] _CUDA_PIPELINE_STATIC_QUALIFIER void
okl_memcpy_async(void *__restrict__ dst_shared,
                      const void *__restrict__ src_global,
                      size_t size_and_align, size_t zfill = 0) {
  __pipeline_memcpy_async(dst_shared, src_global, size_and_align);
}

[[maybe_unused]] _CUDA_PIPELINE_STATIC_QUALIFIER void okl_pipeline_commit() {
   __pipeline_commit();
}

[[maybe_unused]] _CUDA_PIPELINE_STATIC_QUALIFIER void
okl_pipeline_wait_prior(size_t N) { __pipeline_wait_prior(N); }
} // namespace

extern "C" __global__
__launch_bounds__(1) void _occa_intrinsic_builtin_0(const float *fVec,
                                                    float *fSum) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      float value = okl_exp10f(fVec[i]);
      atomicAdd(&(*fSum), value);
    }
  }
}
