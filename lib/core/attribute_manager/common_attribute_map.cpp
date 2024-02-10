#include "oklt/core/attribute_manager/common_attribute_map.h"
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/pipeline/stages/transpiler/error_codes.h>

namespace oklt {
using namespace clang;

bool CommonAttributeMap::registerHandler(std::string name, AttrDeclHandler handler) {
  auto ret = _declHandlers.insert(std::make_pair(std::move(name), std::move(handler)));
  return ret.second;
}

bool CommonAttributeMap::registerHandler(std::string name, AttrStmtHandler handler) {
  auto ret = _stmtHandlers.insert(std::make_pair(std::move(name), std::move(handler)));
  return ret.second;
}

bool CommonAttributeMap::handleAttr(const Attr* attr,
                                    const Decl* decl,
                                    SessionStage& stage)
{
  std::string name = attr->getNormalizedFullName();
  auto it = _declHandlers.find(name);
  if (it == _declHandlers.end()) {
    std::string description = "Common attribute handler for declaration is missing";
    stage.pushError(make_error_code(OkltTranspilerErrorCode::ATTRIBUTE_HANDLER_IS_MISSING), description);
    return false;
  }
  return it->second.handle(attr, decl, stage);
}

bool CommonAttributeMap::handleAttr(const Attr* attr,
                                    const Stmt* stmt,
                                    SessionStage& stage)
{
  std::string name = attr->getNormalizedFullName();
  auto it = _stmtHandlers.find(name);
  if (it == _stmtHandlers.end()) {
    std::string description = "Common attribute handler for statement is missing";
    stage.pushError(make_error_code(OkltTranspilerErrorCode::ATTRIBUTE_HANDLER_IS_MISSING), description);
    return false;
  }
  return it->second.handle(attr, stmt, stage);
}

bool CommonAttributeMap::hasAttrHandler(const std::string& name) const {
  auto declIt = _declHandlers.find(name);
  if (declIt != _declHandlers.cend()) {
    return true;
  }
  auto stmtIt = _stmtHandlers.find(name);
  return stmtIt != _stmtHandlers.cend();
}

}  // namespace oklt
