#pragma once

#include "oklt/core/attribute_manager/common_attribute_map.h"
#include "oklt/core/attribute_manager/backend_attribute_map.h"

#include <llvm/Support/Error.h>
#include <string>

namespace oklt {

class AttributeManager {
protected:
  AttributeManager() = default;
  ~AttributeManager() = default;
public:

  AttributeManager(const AttributeManager &) = delete;
  AttributeManager(AttributeManager &&) = delete;
  AttributeManager & operator = (const AttributeManager &) = delete;
  AttributeManager & operator = (AttributeManager &&) = delete;

  static AttributeManager &instance();

  bool registerCommonHandler(std::string name, AttrDeclHandler handler);
  bool registerCommonHandler(std::string name, AttrStmtHandler handler);

  bool registerBackendHandler(BackendAttributeMap::KeyType key, AttrDeclHandler handler);
  bool registerBackendHandler(BackendAttributeMap::KeyType key, AttrStmtHandler handler);

  bool handleAttr(const clang::Attr* attr, const clang::Decl *decl, SessionStage &session);
  bool handleAttr(const clang::Attr* attr, const clang::Stmt *stmt, SessionStage &session);

  llvm::Expected<const clang::Attr*> checkAttrs(const clang::AttrVec &attrs,
                                          const clang::Decl *decl,
                                          SessionStage &session);
  llvm::Expected<const clang::Attr*> checkAttrs(const clang::ArrayRef<const clang::Attr*> &attrs,
                                           const clang::Stmt *decl,
                                           SessionStage &session);
private:
  //INFO: here should not be the same named attributes in both
  //      might need to handle uniqueness

  //INFO: if build AttributeViwer just wrap into shared_ptr and copy it
  CommonAttributeMap _commonAttrs;
  BackendAttributeMap _backendAttrs;
};
}
