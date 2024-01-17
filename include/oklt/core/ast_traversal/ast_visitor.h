#pragma once

#include "clang/AST/RecursiveASTVisitor.h"
#include <memory>

namespace oklt {

class TranspileSession;

class ASTVisitor : public clang::RecursiveASTVisitor<ASTVisitor> {
public:
  explicit ASTVisitor(TranspileSession &session);
  virtual ~ASTVisitor() = default;

  bool TraverseDecl(clang::Decl *decl);
  bool TraverseStmt(clang::Stmt *stmt, DataRecursionQueue *queue = nullptr);

  //INFO: must be public also
  bool VisitFunctionDecl(clang::FunctionDecl *funcDecl);
  bool VisitVarDecl(clang::VarDecl *varDecl);

protected:
  TranspileSession &_session;
};

}
