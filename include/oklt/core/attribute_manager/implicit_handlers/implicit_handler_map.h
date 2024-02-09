#pragma once

#include <oklt/core/attribute_manager/implicit_handlers/decl_handler.h>
#include <oklt/core/attribute_manager/implicit_handlers/stmt_handler.h>
#include <oklt/core/target_backends.h>

#include <map>
#include <tuple>

namespace oklt {
class ImplicitHandlerMap {
 public:
  using KeyType = std::tuple<TargetBackend, int>;
  using DeclHandlers = std::map<KeyType, DeclHandler>;
  using StmtHandlers = std::map<KeyType, StmtHandler>;

  ImplicitHandlerMap() = default;
  ~ImplicitHandlerMap() = default;

  bool registerHandler(KeyType key, DeclHandler handler);
  bool registerHandler(KeyType key, StmtHandler handler);

  bool operator()(const clang::Decl* decl, SessionStage& stage);
  bool operator()(const clang::Stmt* stmt, SessionStage& stage);

 private:
  DeclHandlers _declHandlers;
  StmtHandlers _stmtHandlers;
};
}  // namespace oklt
