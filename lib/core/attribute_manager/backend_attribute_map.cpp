#include "oklt/core/attribute_manager/backend_attribute_map.h"
#include "oklt/core/transpiler_session/session_stage.h"
#include <oklt/pipeline/stages/transpiler/error_codes.h>

namespace oklt {
using namespace clang;

bool BackendAttributeMap::registerHandler(KeyType key, AttrDeclHandler handler) {
  auto ret = _declHandlers.insert(std::make_pair(std::move(key), std::move(handler)));
  return ret.second;
}

bool BackendAttributeMap::registerHandler(KeyType key, AttrStmtHandler handler) {
  auto ret = _stmtHandlers.insert(std::make_pair(std::move(key), std::move(handler)));
  return ret.second;
}

bool BackendAttributeMap::handleAttr(const Attr* attr,
                                     const Decl* decl,
                                     SessionStage& stage)
{
  std::string name = attr->getNormalizedFullName();
  auto backend = stage.getBackend();
  auto it = _declHandlers.find(std::make_tuple(backend, name));
  if (it == _declHandlers.end()) {
    std::string description = "Backend attribute handler for declaration is missing";
    stage.pushError(make_error_code(OkltTranspilerErrorCode::ATTRIBUTE_HANDLER_IS_MISSING), description);
    return false;
  }
  return it->second.handle(attr, decl, stage);
}

bool BackendAttributeMap::handleAttr(const Attr* attr,
                                     const Stmt* stmt,
                                     SessionStage& stage)
{
  std::string name = attr->getNormalizedFullName();
  auto backend = stage.getBackend();
  auto it = _stmtHandlers.find(std::make_tuple(backend, name));
  if (it != _stmtHandlers.end()) {
    std::string description = "Backend attribute handler for statement is missing";
    stage.pushError(make_error_code(OkltTranspilerErrorCode::ATTRIBUTE_HANDLER_IS_MISSING), description);
    return false;
  }
  return it->second.handle(attr, stmt, stage);
}

bool BackendAttributeMap::hasAttrHandler(SessionStage& stage, const std::string& name) const {
  auto key = std::make_tuple(stage.getBackend(), name);
  auto declIt = _declHandlers.find(key);
  if (declIt != _declHandlers.cend()) {
    return true;
  }
  auto stmtIt = _stmtHandlers.find(key);
  return stmtIt != _stmtHandlers.cend();
}
}  // namespace oklt
