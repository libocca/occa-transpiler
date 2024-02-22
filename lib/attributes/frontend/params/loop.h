#pragma once

#include <clang/AST/Attr.h>
#include "core/transpiler_session/session_stage.h"

namespace oklt {

enum class LoopType {
    Regular,
    Inner,
    Outer,
};

enum class Dim {
    X = 0,
    Y = 1,
    Z = 2,
    Auto = 3,
};

struct AttributedLoop {
    LoopType type = LoopType::Regular;
    Dim dim = Dim::Auto;
};

}  // namespace oklt
