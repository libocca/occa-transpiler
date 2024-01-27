#pragma once

#include <memory>
#include "clang/AST/RecursiveASTVisitor.h"

namespace oklt {

class SessionStage;

class ASTVisitor : public clang::RecursiveASTVisitor<ASTVisitor> {
 public:
  explicit ASTVisitor(SessionStage& session);
  virtual ~ASTVisitor() = default;

  bool TraverseDecl(clang::Decl* decl);
  bool TraverseStmt(clang::Stmt* stmt, DataRecursionQueue* queue = nullptr);
  bool TraverseRecoveryExpr(clang::RecoveryExpr* expr, DataRecursionQueue* queue = nullptr);

 protected:
  SessionStage& _session;
};

}  // namespace oklt
