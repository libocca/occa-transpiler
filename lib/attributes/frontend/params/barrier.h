#pragma once

#include <clang/AST/Attr.h>
#include "core/transpiler_session/session_stage.h"

namespace oklt {

enum class BarrierType {
    syncDefault,
    syncWarp,
};

struct AttributedBarrier {
    BarrierType type = BarrierType::syncDefault;
};

}  // namespace oklt
