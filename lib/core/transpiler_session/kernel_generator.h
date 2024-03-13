#pragma once

#include <string>
#include <tl/expected.hpp>

namespace oklt {
class SessionStage;
struct Error;

tl::expected<std::string, Error> generateTranspiledKernel(SessionStage& stage);
}  // namespace oklt
