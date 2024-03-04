#pragma once
#include <clang/AST/Attr.h>

namespace oklt {

enum class AttributedLoopType {
    Regular,
    Inner,
    Outer,
};

enum class DimType {
    X = 0,
    Y = 1,
    Z = 2,
    Auto = 3,
};

struct AttributedLoop {
    AttributedLoopType type = AttributedLoopType::Regular;
    DimType dim = DimType::Auto;
};

}  // namespace oklt
