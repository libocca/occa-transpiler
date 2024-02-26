#pragma once

#include "attributes/frontend/params/loop.h"
#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>

namespace oklt {
ParseResult parseLoopAttrParams(const clang::Attr* a, SessionStage& s, LoopType loopType);
}
