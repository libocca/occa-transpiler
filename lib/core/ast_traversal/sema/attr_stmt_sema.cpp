#include <oklt/core/ast_traversal/sema/attr_stmt_sema.h>
#include <oklt/core/ast_traversal/transpile_types/function_info.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt {

bool AttrStmtSema::beforeTraverse(clang::AttributedStmt* stmt, SessionStage& stage) {
  return true;
}

bool AttrStmtSema::afterTraverse(clang::AttributedStmt* stmt, SessionStage& stage) {
  return true;
}

}  // namespace oklt
