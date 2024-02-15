#pragma once

#include <string>
#include <tl/expected.hpp>

namespace oklt {

enum struct AstProcessorType : unsigned char {
    OKL_NO_SEMA,
    OKL_WITH_SEMA,
};

tl::expected<AstProcessorType, std::string> AstProcessorFromString(const std::string& type);
std::string processorToString(AstProcessorType proc_type);
}  // namespace oklt
