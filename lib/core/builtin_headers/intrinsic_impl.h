#pragma once
#include <string>
#include "oklt/core/target_backends.h"

namespace oklt {

// constexpr const char

std::string getIntrinsicIncSource(TargetBackend backend);
}
