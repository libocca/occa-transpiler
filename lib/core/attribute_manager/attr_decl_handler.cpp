#include "oklt/core/attribute_manager/attr_decl_handler.h"
#include "oklt/core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

AttrDeclHandler::AttrDeclHandler(ParamsParserType pp, HandleType h)
    : _paramsParser(std::move(pp)), _handler(std::move(h)) {}

bool AttrDeclHandler::handle(const Attr* attr,
                             const Decl* decl,
                             SessionStage& stage,
                             HandledChanges callback)
{
  if (parseParams(attr, stage)) {
    return _handler(attr, decl, stage, callback);
  }
  return false;
}

bool AttrDeclHandler::parseParams(const Attr* attr, SessionStage& stage) {
  return _paramsParser(attr, stage);
}
}  // namespace oklt
