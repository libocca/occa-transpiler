namespace {
//Math functions

//Single presicion
inline __device__ float okl_exp10f (float x) {
    return exp10f(x);
}

// Warp Shuffle Functions

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
