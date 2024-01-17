#pragma once

#include "oklt/core/config.h"

#include <filesystem>
#include <iosfwd>

namespace okl {
bool transpile(std::ostream &error_stream,
               const std::filesystem::path &source_file,
               const std::filesystem::path &output_file,
               TRANSPILER_TYPE targetBackend,
               bool need_normalization = true);
}

