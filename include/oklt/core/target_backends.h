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

    _LAUNCHER,
};

tl::expected<TargetBackend, std::string> backendFromString(const std::string& type);
std::string backendToString(TargetBackend backend);
bool isHostCategory(TargetBackend backend);
bool isDeviceCategory(TargetBackend backend);

}  // namespace oklt
