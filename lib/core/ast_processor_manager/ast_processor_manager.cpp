#include <oklt/core/ast_processor_manager/ast_processor_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt {
using namespace clang;

AstProcessorManager& AstProcessorManager::instance() {
  static AstProcessorManager manager;
  return manager;
}

bool AstProcessorManager::registerGenericHandle(AstProcessorType procType, DeclNodeHandle handle) {
  auto [_, ret] = _genericDeclHandle.try_emplace(procType, std::move(handle));
  return ret;
}

bool AstProcessorManager::registerGenericHandle(AstProcessorType procType, StmtNodeHandle handle) {
  auto [_, ret] = _genericStmtHandle.try_emplace(procType, std::move(handle));
  return ret;
}

bool AstProcessorManager::registerSpecificNodeHandle(KeyType key, DeclNodeHandle handler) {
  auto [_, ret] = _declHandlers.try_emplace(std::move(key), std::move(handler));
  return ret;
}

bool AstProcessorManager::registerSpecificNodeHandle(KeyType key, StmtNodeHandle handler) {
  auto [_, ret] = _stmtHandlers.try_emplace(std::move(key), std::move(handler));
  return ret;
}

bool AstProcessorManager::runPreActionNodeHandle(AstProcessorType procType,
                                                 const Decl* decl,
                                                 SessionStage& stage) {
  auto it = _declHandlers.find({procType, decl->getKind()});
  if (it == _declHandlers.end()) {
    auto gen_it = _genericDeclHandle.find(procType);
    if (gen_it == _genericDeclHandle.end()) {
      return true;
    }
    return gen_it->second.preAction(decl, stage);
  }
  return it->second.preAction(decl, stage);
}

bool AstProcessorManager::runPostActionNodeHandle(AstProcessorType procType,
                                                  const Decl* decl,
                                                  SessionStage& stage) {
  auto it = _declHandlers.find({procType, decl->getKind()});
  if (it == _declHandlers.end()) {
    auto gen_it = _genericDeclHandle.find(procType);
    if (gen_it == _genericDeclHandle.end()) {
      return true;
    }
    return gen_it->second.postAction(decl, stage);
  }
  return it->second.postAction(decl, stage);
}

bool AstProcessorManager::runPreActionNodeHandle(AstProcessorType procType,
                                                 const Stmt* stmt,
                                                 SessionStage& stage) {
  auto it = _stmtHandlers.find({procType, stmt->getStmtClass()});
  if (it == _stmtHandlers.end()) {
    auto gen_it = _genericStmtHandle.find(procType);
    if (gen_it == _genericStmtHandle.end()) {
      return true;
    }
    return gen_it->second.preAction(stmt, stage);
  }
  return it->second.preAction(stmt, stage);
}

bool AstProcessorManager::runPostActionNodeHandle(AstProcessorType procType,
                                                  const Stmt* stmt,
                                                  SessionStage& stage) {
  auto it = _stmtHandlers.find({procType, stmt->getStmtClass()});
  if (it == _stmtHandlers.end()) {
    auto gen_it = _genericStmtHandle.find(procType);
    if (gen_it == _genericStmtHandle.end()) {
      return true;
    }
    return gen_it->second.postAction(stmt, stage);
  }
  return it->second.postAction(stmt, stage);
}
}  // namespace oklt
