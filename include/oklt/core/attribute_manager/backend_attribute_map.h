#pragma once

#include <map>
#include <tuple>
#include "oklt/core/attribute_manager/attr_decl_handler.h"
#include "oklt/core/attribute_manager/attr_stmt_handler.h"
#include "oklt/core/config.h"

namespace oklt {
class BackendAttributeMap {
 public:
  using KeyType = std::tuple<TRANSPILER_TYPE, std::string>;
  using DeclHandlers = std::map<KeyType, AttrDeclHandler>;
  using StmtHandlers = std::map<KeyType, AttrStmtHandler>;

  BackendAttributeMap() = default;
  ~BackendAttributeMap() = default;

  bool registerHandler(KeyType key, AttrDeclHandler handler);
  bool registerHandler(KeyType key, AttrStmtHandler handler);

  bool handleAttr(const clang::Attr* attr, const clang::Decl* decl, SessionStage& stage);
  bool handleAttr(const clang::Attr* attr, const clang::Stmt* stmt, SessionStage& stage);

  bool hasAttrHandler(SessionStage& stage, const std::string& name);

 private:
  DeclHandlers _declHandlers;
  StmtHandlers _stmtHandlers;
};
}  // namespace oklt
