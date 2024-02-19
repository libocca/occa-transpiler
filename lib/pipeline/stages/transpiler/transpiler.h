#pragma once

#include "core/transpiler_session/session_result.h"
#include "core/transpiler_session/transpiler_session.h"

namespace oklt {
TranspilerSessionResult runTranspilerStage(SharedTranspilerSession session);
}  // namespace oklt
