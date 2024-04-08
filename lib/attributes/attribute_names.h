#pragma once
#include <clang/Basic/Version.h>

namespace oklt {
constexpr const char KERNEL_ATTR_NAME[] = "okl_kernel";
constexpr const char OUTER_ATTR_NAME[] = "okl_outer";
constexpr const char INNER_ATTR_NAME[] = "okl_inner";
constexpr const char TILE_ATTR_NAME[] = "okl_tile";
constexpr const char SHARED_ATTR_NAME[] = "okl_shared";
constexpr const char DIM_ATTR_NAME[] = "okl_dim";
constexpr const char DIM_ORDER_ATTR_NAME[] = "okl_dimOrder";
constexpr const char RESTRICT_ATTR_NAME[] = "okl_restrict";
constexpr const char BARRIER_ATTR_NAME[] = "okl_barrier";
constexpr const char NO_BARRIER_ATTR_NAME[] = "okl_nobarrier";
constexpr const char EXCLUSIVE_ATTR_NAME[] = "okl_exclusive";
constexpr const char ATOMIC_ATTR_NAME[] = "okl_atomic";
constexpr const char MAX_INNER_DIMS[] = "okl_max_inner_dims";

}  // namespace oklt
