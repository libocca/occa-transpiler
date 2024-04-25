#pragma once
#include <string>
#include "oklt/core/target_backends.h"

namespace oklt {

constexpr const char INTRINSIC_INCLUDE_FILENAME[] = "./okl_intrinsic.h";
std::string getIntrinsicIncSource(TargetBackend backend);
}
