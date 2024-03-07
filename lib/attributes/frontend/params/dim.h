#pragma once

#include <clang/AST/Attr.h>
#include "core/transpiler_session/session_stage.h"

namespace oklt {

struct AttributedDimOrder {
    std::vector<size_t> idx = {};
};

struct AttributedDim {
    std::vector<std::string> dim = {};
};

}  // namespace oklt
