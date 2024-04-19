#pragma once

#include <oklt/core/error.h>
#include <tl/expected.hpp>

#include <any>

namespace oklt {
struct Error;

using HandleResult = tl::expected<std::any, Error>;

}  // namespace oklt
