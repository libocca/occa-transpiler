#pragma once

#include <clang/AST/RecursiveASTVisitor.h>

namespace oklt {

class SessionStage;
class AstProcessorManager;

class PreorderNlrTraversal : public clang::RecursiveASTVisitor<PreorderNlrTraversal> {
 public:
  explicit PreorderNlrTraversal(AstProcessorManager& procMng, SessionStage& stage);
  bool TraverseDecl(clang::Decl* decl);
  bool TraverseStmt(clang::Stmt* stmt);
  bool TraverseRecoveryExpr(clang::RecoveryExpr* recoveryExpr);
  bool TraverseTranslationUnitDecl(clang::TranslationUnitDecl* translationUnitDecl);

 private:
  AstProcessorManager& _procMng;
  SessionStage& _stage;
};

}  // namespace oklt
