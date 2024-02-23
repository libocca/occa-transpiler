#pragma once

#include <clang/AST/Attr.h>
#include "attributes/frontend/params/loop.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
tl::expected<std::any, Error> parseLoopAttrParams(const clang::Attr* a, SessionStage& s, LoopType loopType);
}
