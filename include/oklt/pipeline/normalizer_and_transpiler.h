#pragma once

#include <oklt/core/transpiler_session/session_result.h>
#include <oklt/core/transpiler_session/transpiler_session.h>

namespace oklt {
TranspilerSessionResult normalizeAndTranspile(SharedTranspilerSession session);
}
