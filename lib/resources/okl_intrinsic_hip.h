namespace {

//Single presicion
[[maybe_unused]]
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
    return __shfl_xor_sync(mask, var, laneMask, width);
}


[[maybe_unused]]
inline __device__ void okl_memcpy_async(void* dst_shared,
                             const void* src_global,
                             size_t size_and_align,
                             size_t zfill=0)
{
    size_t i = 0;
    // copy
    for (; i < size_and_align - zfill; ++i) {
        ((char*)dst_shared)[i] = ((char*)src_global)[i];
    }
    // zero-fill
    for (; i < size_and_align; ++i) {
        ((char*)dst_shared)[i] = 0;
    }
}

[[maybe_unused]]
inline __device__ void okl_pipeline_commit() {
}

[[maybe_unused]]
inline __device__ void okl_pipeline_wait_prior(size_t N) {
}
}
