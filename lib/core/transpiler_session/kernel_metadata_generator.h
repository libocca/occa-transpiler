#pragma once

#include <string>
#include <tl/expected.hpp>

namespace oklt{
struct Error;
class SessionStage;

tl::expected<std::string, Error> generateKernelMetaData(SessionStage& stage);
}
