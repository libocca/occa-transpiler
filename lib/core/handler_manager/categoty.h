#pragma once

#include <tl/expected.hpp>

namespace oklt {
enum struct HandlerCategory : unsigned char {
    BACKEND,
    PARSER,
    SEMA,
};

//tl::expected<TargetBackend, std::string> backendFromString(const std::string& type);
//std::string backendToString(TargetBackend backend);
//bool isHostCategory(TargetBackend backend);
//bool isDeviceCategory(TargetBackend backend);

}  // namespace oklt
