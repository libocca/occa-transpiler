#pragma once

#include <filesystem>
#include <ostream>

#include "tl/expected.hpp"

namespace oklt {
enum struct TargetBackend : unsigned char {
    SERIAL,
    OPENMP,
    CUDA,
    HIP,
    DPCPP,
};

tl::expected<TargetBackend, std::string> backendFromString(const std::string& type);
std::string backendToString(TargetBackend backend);
}  // namespace oklt
