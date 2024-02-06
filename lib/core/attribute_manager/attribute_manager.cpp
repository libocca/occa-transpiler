#include "oklt/core/attribute_manager/attribute_manager.h"
#include "oklt/core/transpiler_session/session_stage.h"

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

bool AttributeManager::handleAttr(const Attr* attr,
                                  const Decl* decl,
                                  SessionStage& stage,
                                  HandledChanges callback)
{
  std::string name = attr->getNormalizedFullName();
  if (_commonAttrs.hasAttrHandler(name)) {
    return _commonAttrs.handleAttr(attr, decl, stage, callback);
  }

  if (_backendAttrs.hasAttrHandler(stage, name)) {
    return _backendAttrs.handleAttr(attr, decl, stage, callback);
  }

  return false;
}

bool AttributeManager::handleAttr(const Attr* attr,
                                  const Stmt* stmt,
                                  SessionStage& stage,
                                  HandledChanges callback)
{
  std::string name = attr->getNormalizedFullName();
  if (_commonAttrs.hasAttrHandler(name)) {
    return _commonAttrs.handleAttr(attr, stmt, stage, callback);
  }
  if (_backendAttrs.hasAttrHandler(stage, name)) {
    return _backendAttrs.handleAttr(attr, stmt, stage, callback);
  }
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
