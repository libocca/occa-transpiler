// Auto generated file.
#pragma once

static constexpr const char INTRINSIC_HOST[] = R"delim(
namespace {
//Math functions
inline  float okl_exp10f(float x) {
    return exp10f(x);
}

inline void okl_memcpy_async(void* dst_shared,
                            const void*  src_global,
                            size_t size_and_align,
                            size_t zfill = 0)
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

inline void okl_pipeline_commit()
{

}

void okl_pipeline_wait_prior(size_t)
{
}
}

)delim";
