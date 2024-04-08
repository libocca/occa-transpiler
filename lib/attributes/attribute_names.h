#pragma once

namespace oklt {
constexpr const char KERNEL_ATTR_NAME[] = "okl::kernel";
constexpr const char OUTER_ATTR_NAME[] = "okl::outer";
constexpr const char INNER_ATTR_NAME[] = "okl::inner";
constexpr const char TILE_ATTR_NAME[] = "okl::tile";
constexpr const char SHARED_ATTR_NAME[] = "okl::shared";
constexpr const char DIM_ATTR_NAME[] = "okl::dim";
constexpr const char DIMORDER_ATTR_NAME[] = "okl::dimOrder";
constexpr const char RESTRICT_ATTR_NAME[] = "okl::restrict";
constexpr const char BARRIER_ATTR_NAME[] = "okl::barrier";
constexpr const char NOBARRIER_ATTR_NAME[] = "okl::nobarrier";
constexpr const char EXCLUSIVE_ATTR_NAME[] = "okl::exclusive";
constexpr const char ATOMIC_ATTR_NAME[] = "okl::atomic";
constexpr const char MAX_INNER_DIMS[] = "okl::max_inner_dims";

const int CXX_ATTRIBUTE_BEGIN_TO_NAME_OFFSET = 2;
const int GNU_ATTRIBUTE_BEGIN_TO_NAME_OFFSET = 15;
}  // namespace oklt
