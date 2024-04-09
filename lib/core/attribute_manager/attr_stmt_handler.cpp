#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

HandleResult AttrStmtHandler::handle(SessionStage& stage,
                                     const clang::Stmt& stmt,
                                     const clang::Attr& attr,
                                     const std::any* params) {
    return _handler(stage, stmt, attr, params);
}

}  // namespace oklt
