#pragma once

#include "llvm/Support/Error.h"
#include <filesystem>

namespace oklt::util {
llvm::Expected<std::string> read_file_as_str(const std::filesystem::path &);
llvm::Error write_file_as_str(const std::filesystem::path &,  std::string_view);
} // namespace okl::util

