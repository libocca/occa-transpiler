#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

HandleResult AttrStmtHandler::handle(const clang::Attr& attr,
                                     const clang::Stmt& stmt,
                                     const std::any* params,
                                     SessionStage& stage) {
    return _handler(attr, stmt, params, stage);
}

}  // namespace oklt
