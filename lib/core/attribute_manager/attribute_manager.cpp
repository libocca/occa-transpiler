#include "oklt/core/attribute_manager/attribute_manager.h"
#include "oklt/core/transpiler_session/session_stage.h"
#include <oklt/pipeline/stages/transpiler/error_codes.h>

namespace oklt {
using namespace clang;

AttributeManager& AttributeManager::instance() {
  static AttributeManager attrManager;
  return attrManager;
}

bool AttributeManager::registerCommonHandler(std::string name, AttrDeclHandler handler) {
  return _commonAttrs.registerHandler(std::move(name), std::move(handler));
}

bool AttributeManager::registerCommonHandler(std::string name, AttrStmtHandler handler) {
  return _commonAttrs.registerHandler(std::move(name), std::move(handler));
}

bool AttributeManager::registerBackendHandler(BackendAttributeMap::KeyType key,
                                              AttrDeclHandler handler) {
  return _backendAttrs.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerBackendHandler(BackendAttributeMap::KeyType key,
                                              AttrStmtHandler handler) {
  return _backendAttrs.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerImplicitHandler(ImplicitHandlerMap::KeyType key,
                                               DeclHandler handler) {
  return _implicitHandlers.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::registerImplicitHandler(ImplicitHandlerMap::KeyType key,
                                               StmtHandler handler) {
  return _implicitHandlers.registerHandler(std::move(key), std::move(handler));
}

bool AttributeManager::handleAttr(const Attr* attr, const Decl* decl, SessionStage& stage) {
  std::string name = attr->getNormalizedFullName();
  if (_commonAttrs.hasAttrHandler(name)) {
    return _commonAttrs.handleAttr(attr, decl, stage);
  }

  if (_backendAttrs.hasAttrHandler(stage, name)) {
    return _backendAttrs.handleAttr(attr, decl, stage);
  }
  std::string description = "Backend attribute handler for declaration is missing";
  stage.pushError(make_error_code(OkltTranspilerErrorCode::ATTRIBUTE_HANDLER_IS_MISSING), description);
  return false;
}

bool AttributeManager::handleStmt(const Stmt* stmt, SessionStage& stage) {
  return _implicitHandlers(stmt, stage);
}

bool AttributeManager::handleDecl(const Decl* decl, SessionStage& stage) {
  return _implicitHandlers(decl, stage);
}

bool AttributeManager::handleAttr(const Attr* attr, const Stmt* stmt, SessionStage& stage) {
  std::string name = attr->getNormalizedFullName();
  if (_commonAttrs.hasAttrHandler(name)) {
    return _commonAttrs.handleAttr(attr, stmt, stage);
  }
  if (_backendAttrs.hasAttrHandler(stage, name)) {
    return _backendAttrs.handleAttr(attr, stmt, stage);
  }
  std::string description = "Backend attribute handler for statement is missing";
  stage.pushError(make_error_code(OkltTranspilerErrorCode::ATTRIBUTE_HANDLER_IS_MISSING), description);
  return false;
}

bool AttributeManager::hasAttrHandler(const clang::Attr *occaAttr, SessionStage& stage) const
{
  auto name = occaAttr->getNormalizedFullName();
  if (_commonAttrs.hasAttrHandler(name)) {
    return true;
  }
  return _backendAttrs.hasAttrHandler(stage, name);
}

}  // namespace oklt
