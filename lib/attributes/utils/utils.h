#pragma once

#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

namespace oklt {
const clang::AttributedStmt* getAttributedStmt(SessionStage& s, const clang::Stmt& stmt);
}
