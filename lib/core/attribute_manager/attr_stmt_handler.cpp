#include "oklt/core/attribute_manager/attr_stmt_handler.h"
#include "oklt/core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;
AttrStmtHandler::AttrStmtHandler(ParamsParserType pp, HandleType h)
    : _paramsParser(std::move(pp)),
      _handler(std::move(h)) {}

bool AttrStmtHandler::handle(const clang::Attr* attr,
                             const clang::Stmt* stmt,
                             SessionStage& stage) {
    if (parseParams(attr, stage)) {
        return _handler(attr, stmt, stage);
    }
    return false;
}

bool AttrStmtHandler::parseParams(const clang::Attr* attr, SessionStage& stage) {
    return _paramsParser(attr, stage);
}
}  // namespace oklt
