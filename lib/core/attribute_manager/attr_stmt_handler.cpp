#include "oklt/core/attribute_manager/attr_stmt_handler.h"
#include "oklt/core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

bool AttrStmtHandler::handle(const clang::Attr* attr,
                             const clang::Stmt* stmt,
                             SessionStage& stage) {
    return _handler(attr, stmt, stage);
}

}  // namespace oklt
