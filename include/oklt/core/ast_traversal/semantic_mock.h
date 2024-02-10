#pragma once

#include <clang/AST/RecursiveASTVisitor.h>
#include <oklt/core/ast_traversal/validate_attributes.h>

namespace oklt {

class SessionStage;

class SemanticMockVisitor :
                          public clang::RecursiveASTVisitor<SemanticMockVisitor>
{
 public:
  explicit SemanticMockVisitor(SessionStage& stage,
                               AttrValidatorFnType attrValidateFn = validateAttributes);
  ~SemanticMockVisitor() = default;

  bool TraverseDecl(clang::Decl* decl);
  bool TraverseStmt(clang::Stmt* stmt, DataRecursionQueue* queue = nullptr);
  bool TraverseRecoveryExpr(clang::RecoveryExpr* expr, DataRecursionQueue* queue = nullptr);
 protected:
  SessionStage& _stage;
  AttrValidatorFnType _attrValidator;
};

}  // namespace oklt
