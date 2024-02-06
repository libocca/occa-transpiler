#pragma once

#include <map>
#include <tuple>
#include "oklt/core/attribute_manager/attr_decl_handler.h"
#include "oklt/core/attribute_manager/attr_stmt_handler.h"
#include "oklt/core/target_backends.h"

namespace oklt {
class BackendAttributeMap {
 public:
  using KeyType = std::tuple<TargetBackend, std::string>;
  using DeclHandlers = std::map<KeyType, AttrDeclHandler>;
  using StmtHandlers = std::map<KeyType, AttrStmtHandler>;

  BackendAttributeMap() = default;
  ~BackendAttributeMap() = default;

  bool registerHandler(KeyType key, AttrDeclHandler handler);
  bool registerHandler(KeyType key, AttrStmtHandler handler);

  bool handleAttr(const clang::Attr* attr,
                  const clang::Decl* decl,
                  SessionStage& stage,
                  HandledChanges callback);
  bool handleAttr(const clang::Attr* attr,
                  const clang::Stmt* stmt,
                  SessionStage& stage,
                  HandledChanges callback);

  [[nodiscard]] bool hasAttrHandler(SessionStage& stage, const std::string& name) const;

 private:
  DeclHandlers _declHandlers;
  StmtHandlers _stmtHandlers;
};
}  // namespace oklt
