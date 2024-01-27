#pragma once

#include <filesystem>
#include <ostream>

#include "tl/expected.hpp"

namespace oklt {
enum struct TRANSPILER_TYPE : unsigned char {
  OPENMP,
  CUDA,
};

tl::expected<TRANSPILER_TYPE, std::string> backendFromString(const std::string& type);
std::string backendToString(TRANSPILER_TYPE backend);
}  // namespace oklt
