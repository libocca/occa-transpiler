#pragma once

#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

namespace oklt {
const clang::AttributedStmt* getAttributedStmt(SessionStage&, const clang::Stmt&);
std::string getCleanTypeString(clang::QualType);
}
