#include "oklt/core/attribute_manager/attr_decl_handler.h"
#include "oklt/core/transpiler_session/transpiler_session.h"

namespace oklt {
using namespace clang;

AttrDeclHandler::AttrDeclHandler(ParamsParserType pp, HandleType h)
    : _paramsParser(std::move(pp)), _handler(std::move(h)) {}

bool AttrDeclHandler::handle(const Attr* attr, const Decl* decl, SessionStage& session) {
  if (parseParams(attr, session)) {
    return _handler(attr, decl, session);
  }
  return false;
}

bool AttrDeclHandler::parseParams(const Attr* attr, SessionStage& session) {
  return _paramsParser(attr, session);
}
}  // namespace oklt
