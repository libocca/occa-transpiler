#pragma once

#include <string>
#include <tl/expected.hpp>

namespace oklt {
struct Transpilation;
struct Error;

tl::expected<std::string, Error> decodeKernelModifier(const Transpilation& t);
tl::expected<std::string, Error> decodeParamModifier(const Transpilation& t);
}  // namespace oklt
