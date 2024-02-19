#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

bool AttrStmtHandler::handle(const clang::Attr* attr,
                             const clang::Stmt* stmt,
                             SessionStage& stage) {
    return _handler(attr, stmt, stage);
}

}  // namespace oklt
