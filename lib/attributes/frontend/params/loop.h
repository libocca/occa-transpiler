#pragma once
#include <oklt/core/kernel_metadata.h>

#include <clang/AST/Attr.h>

namespace oklt {

enum class Axis {
    X = 0,
    Y = 1,
    Z = 2,
    Auto = 3,
};

struct AttributedLoop {
    LoopType type = LoopType::Regular;
    Axis axis = Axis::Auto;
};

}  // namespace oklt
