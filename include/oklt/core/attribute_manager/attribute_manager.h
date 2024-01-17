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

  void registerHandler(std::string &&name, AttrDeclHandler &&handler);
  void registerHandler(std::string &&name, AttrStmtHandler &&handler);

  void registerHandler(BackendAttributeMap::KeyType &&key, AttrDeclHandler &&handler);
  void registerHandler(BackendAttributeMap::KeyType &&key, AttrStmtHandler &&handler);

  bool handleAttr(const clang::Attr* attr, const clang::Decl *decl, TranspileSession &session);
  bool handleAttr(const clang::Attr* attr, const clang::Stmt *stmt, TranspileSession &session);

  llvm::Expected<const clang::Attr*> checkAttrs(const clang::AttrVec &attrs,
                                          const clang::Decl *decl,
                                          TranspileSession &session);
  llvm::Expected<const clang::Attr*> checkAttrs(const clang::ArrayRef<const clang::Attr*> &attrs,
                                           const clang::Stmt *decl,
                                           TranspileSession &session);
private:
  //INFO: here should not be the same named attributes in both
  //      might need to handle uniqueness
  CommonAttributeMap _commonAttrs;
  BackendAttributeMap _backendAttrs;
};
}
