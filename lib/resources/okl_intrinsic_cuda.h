namespace {
//Math functions

//Single presicion
inline __device__ float okl_exp10f (float x) {
    return exp10f(x);
}

// Warp Shuffle Functions
template<class T>
inline __device__ T okl_shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
{
    return __shfl_sync(mask, var, srcLane, width);
}

template<class T>
inline __device__ T okl_shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)
{
    return __shfl_up_sync(mask, var, delta, width);
}

template<class T>
inline __device__ T okl_shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)
{
    return __shfl_down_sync(mask, var, delta, width);
}

template<class T>
inline __device__
T okl_shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize)
{
    return __shfl_xor_sync(mask, laneMask, width);
}

// Pipeline Primitives Interface
_CUDA_PIPELINE_STATIC_QUALIFIER
void okl_memcpy_async(void* __restrict__ dst_shared,
                      const void* __restrict__ src_global,
                      size_t size_and_align,
                      size_t zfill=0)
{
    __pipeline_memcpy_async(dst_shared, src_global, size_and_align);
}

_CUDA_PIPELINE_STATIC_QUALIFIER
void okl_pipeline_commit() {
    __pipeline_commit();
}

_CUDA_PIPELINE_STATIC_QUALIFIER
 void okl_pipeline_wait_prior(size_t N) {
    __pipeline_wait_prior(N);
}
 }
