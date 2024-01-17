#include "oklt/core/attribute_manager/attr_stmt_handler.h"
#include "oklt/core/transpile_session/transpile_session.h"

namespace oklt {
using namespace clang;
AttrStmtHandler::AttrStmtHandler(ParamsParserType pp, HandleType h)
  :_paramsParser(std::move(pp))
  , _handler(std::move(h))
{}

bool AttrStmtHandler::handle(const clang::Attr *attr,
                             const clang::Stmt *stmt,
                             TranspileSession &session)
{
  if(parseParams(attr, session)) {
    return _handler(attr, stmt, session);
  }
  return false;
}

bool AttrStmtHandler::parseParams(const clang::Attr *attr, TranspileSession &session)
{
  return _paramsParser(attr, session);
}
}
