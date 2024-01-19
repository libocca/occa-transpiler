#pragma once

#include <ostream>
#include <filesystem>

enum struct TRANSPILER_TYPE: unsigned char {
    OPENMP,
    CUDA,
};
