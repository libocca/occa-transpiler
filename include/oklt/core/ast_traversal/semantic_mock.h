#pragma once

#include <memory>
#include "clang/AST/RecursiveASTVisitor.h"
#include <oklt/core/ast_traversal/semantic_base.h>

namespace oklt {

class SessionStage;

class SemanticMockVisitor :
                          public clang::RecursiveASTVisitor<SemanticMockVisitor>
                          , public SemanticASTVisitorBase
{
 public:
  explicit SemanticMockVisitor(SessionStage& stage);
  ~SemanticMockVisitor() override = default;

  bool traverseTranslationUnit(clang::Decl* decl) override;

  bool TraverseDecl(clang::Decl* decl);
  bool TraverseStmt(clang::Stmt* stmt, DataRecursionQueue* queue = nullptr);
  bool TraverseRecoveryExpr(clang::RecoveryExpr* expr, DataRecursionQueue* queue = nullptr);

};

}  // namespace oklt
