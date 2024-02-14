#include <oklt/core/attribute_manager/implicit_handlers/stmt_handler.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt {
using namespace clang;

StmtHandler::StmtHandler(HandleType h)
    : _handler(std::move(h)) {}

bool StmtHandler::operator()(const clang::Stmt* stmt, SessionStage& stage) {
    return _handler(stmt, stage);
}

}  // namespace oklt
