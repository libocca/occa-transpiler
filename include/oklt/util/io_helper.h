#pragma once

#include <filesystem>
#include <tl/expected.hpp>

namespace oklt::util {
tl::expected<std::string, int> readFileAsStr(const std::filesystem::path&);
tl::expected<void, int> writeFileAsStr(const std::filesystem::path&, std::string_view);
}  // namespace oklt::util
