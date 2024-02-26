#include "core/attribute_manager/implicit_handlers/decl_handler.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

DeclHandler::DeclHandler(HandleType h)
    : _handler(std::move(h)) {}

HandleResult DeclHandler::operator()(const Decl* decl, SessionStage& stage) {
    return _handler(decl, stage);
}
}  // namespace oklt
