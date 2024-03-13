#pragma once

#include <string>
#include <tl/expected.hpp>

namespace oklt {
class SessionStage;
struct Error;

tl::expected<std::string, Error> generateTranspiledCode(SessionStage& stage);
tl::expected<std::string, Error> generateTranspiledCodeMetaData(SessionStage& stage);
}  // namespace oklt
