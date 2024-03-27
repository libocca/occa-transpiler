#pragma once

#include <string>
#include <tl/expected.hpp>

namespace oklt {
enum struct TargetBackend {
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

template <>
struct std::hash<oklt::TargetBackend> {
    size_t operator()(const oklt::TargetBackend& t) const noexcept { return size_t(t); }
};
