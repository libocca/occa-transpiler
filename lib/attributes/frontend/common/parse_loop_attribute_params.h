#pragma once

#include <clang/AST/Attr.h>
#include "attributes/frontend/params/loop.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
bool parseLoopAttrParams(const clang::Attr* a, SessionStage& s, LoopType loopType);
}
