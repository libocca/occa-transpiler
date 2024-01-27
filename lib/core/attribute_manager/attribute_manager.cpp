#include "oklt/core/attribute_manager/attribute_manager.h"
#include "oklt/core/transpiler_session/transpiler_session.h"

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

bool AttributeManager::handleAttr(const Attr* attr, const Decl* decl, SessionStage& session) {
  std::string name = attr->getNormalizedFullName();
  if (_commonAttrs.hasAttrHandler(name)) {
    return _commonAttrs.handleAttr(attr, decl, session);
  }

  if (_backendAttrs.hasAttrHandler(session, name)) {
    return _backendAttrs.handleAttr(attr, decl, session);
  }

  return false;
}

bool AttributeManager::handleAttr(const Attr* attr, const Stmt* stmt, SessionStage& session) {
  std::string name = attr->getNormalizedFullName();
  if (_commonAttrs.hasAttrHandler(name)) {
    return _commonAttrs.handleAttr(attr, stmt, session);
  }
  if (_backendAttrs.hasAttrHandler(session, name)) {
    return _backendAttrs.handleAttr(attr, stmt, session);
  }
  return false;
}

llvm::Expected<const clang::Attr*> AttributeManager::checkAttrs(const AttrVec& attrs,
                                                                const Decl* decl,
                                                                SessionStage& session) {
  std::list<Attr*> collectedAttrs;
  for (auto& attr : attrs) {
    auto name = attr->getNormalizedFullName();
    if (_commonAttrs.hasAttrHandler(name)) {
      collectedAttrs.push_back(attr);
      continue;
    }
    if (_backendAttrs.hasAttrHandler(session, name)) {
      collectedAttrs.push_back(attr);
      continue;
    }
  }
  // INFO: there are no OKL attributes at all
  //       might need better solution for this
  if (collectedAttrs.empty()) {
    return llvm::Expected<const clang::Attr*>{nullptr};
  }
  if (collectedAttrs.size() > 1) {
    return llvm::createStringError(std::error_code(), std::string{"Multiple attributes are used"});
  }
  const Attr* attr = collectedAttrs.front();
  return attr;
}

llvm::Expected<const Attr*> AttributeManager::checkAttrs(const ArrayRef<const Attr*>& attrs,
                                                         const Stmt* decl,
                                                         SessionStage& session) {
  std::list<const Attr*> collectedAttrs;
  for (auto& attr : attrs) {
    auto name = attr->getNormalizedFullName();
    if (_commonAttrs.hasAttrHandler(name)) {
      collectedAttrs.push_back(attr);
      continue;
    }
    if (_backendAttrs.hasAttrHandler(session, name)) {
      collectedAttrs.push_back(attr);
      continue;
    }
  }
  // INFO: there are no OKL attributes at all
  //       might need better solution for this
  if (collectedAttrs.empty()) {
    return llvm::Expected<const clang::Attr*>{nullptr};
  }
  if (collectedAttrs.size() > 1) {
    return llvm::createStringError(std::error_code(), std::string{"Multiple attributes are used"});
  }
  const Attr* attr = collectedAttrs.front();
  return attr;
}
}  // namespace oklt
