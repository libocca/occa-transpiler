#pragma once

#include <ostream>
#include <filesystem>

enum struct TRANSPILER_TYPE: unsigned char {
    OPENMP,
    CUDA,
};

struct TranspilerConfig {
  TRANSPILER_TYPE backendType;
  std::filesystem::path sourceFilePath;
  std::ostream &transpiledOutput;
};
