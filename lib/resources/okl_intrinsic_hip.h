#pragma once

#include <stdint.h>

inline void okl_memcpy_async(void* dst_shared,
                             const void* src_global,
                             size_t size_and_align,
                             size_t zfill=0)
{
    //TODO: add impl
}

inline void okl_pipeline_commit() {
    //__pipeline_commit();
}

inline void __pipeline_wait_prior(size_t N) {
    //__pipeline_wait_prior(N);
}
