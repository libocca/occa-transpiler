namespace {
//Math functions
[[maybe_unused]]
inline  float okl_exp10f(float x) {
    return exp10f(x);
}

// Warp Shuffle Functions
template<class T>
inline T okl_shfl_sync(unsigned mask, T var, int srcLane, int width)
{
    return T();
}

template<class T>
inline T okl_shfl_up_sync(unsigned mask, T var, unsigned int delta, int width)
{
    return T();
}

template<class T>
inline T okl_shfl_down_sync(unsigned mask, T var, unsigned int delta, int width)
{
    return T();
}

inline void okl_memcpy_async(void* dst_shared,
                            const void*  src_global,
                            size_t size_and_align,
                            size_t zfill = 0)
{
    std::memcpy(dst_shared, src_global, size_and_align - zfill);
    if(zfill) {
        std::memset(&(((char *)dst_shared)[size_and_align - zfill]), 0, zfill);
    }
}

inline void okl_pipeline_commit()
{

}

void okl_pipeline_wait_prior(size_t)
{

}
}
