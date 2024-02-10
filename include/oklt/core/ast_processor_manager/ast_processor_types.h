#pragma once

#include <string>
#include <tl/expected.hpp>

namespace oklt {

enum struct AstProcessorType : unsigned char {
  OKL_PROGRAM_PROCESSOR_WITHOUT_SEMA,
  OKL_PROGRAM_PROCESSOR_WITH_SEMA,
};

tl::expected<AstProcessorType, std::string> AstProcessorFromString(const std::string& type);
std::string processorToString(AstProcessorType proc_type);
}
