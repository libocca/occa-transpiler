#include "oklt/core/ast_traversal/ast_visitor.h"
#include "oklt/core/transpiler_session/transpiler_session.h"
#include "oklt/core/attribute_manager/attribute_manager.h"

namespace oklt {
using namespace clang;

ASTVisitor::ASTVisitor(SessionStage &session)
    :_session(session)
{}

bool ASTVisitor::TraverseDecl(Decl *decl) {
  if(!decl->hasAttrs()) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseDecl(decl);
  }
  auto &attrManager = _session.getAttrManager();
  llvm::Expected<const Attr*> expectedAttr = attrManager.checkAttrs(decl->getAttrs(),
                                                                    decl,
                                                                    _session);
  if(!expectedAttr) {
    // auto &errorReporter = _session.getErrorReporter();
    auto errorDescription = toString(expectedAttr.takeError());
    // errorReporter.emitError(funcDecl->getSourceRange(),errorDescription);
    return false;
  }
  const Attr* attr = expectedAttr.get();
  //INFO: no OKL attributes to process, continue
  if(!attr) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseDecl(decl);
  }
  return attrManager.handleAttr(attr, decl, _session);
}

bool ASTVisitor::TraverseStmt(Stmt *stmt, DataRecursionQueue *queue) {
  if(!stmt) {
    return true;
  }
  if (stmt->getStmtClass() != Stmt::AttributedStmtClass) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseStmt(stmt);
  }
  AttributedStmt *attrStmt = cast<AttributedStmt>(stmt);
  const Stmt *subStmt = attrStmt->getSubStmt();
  auto &attrManager = _session.getAttrManager();
  llvm::Expected<const Attr*> expectedAttr = attrManager.checkAttrs(attrStmt->getAttrs(),
                                                               stmt,
                                                               _session);
  if(!expectedAttr) {
    // auto &errorReporter = _session.getErrorReporter();
    auto errorDescription = toString(expectedAttr.takeError());
    // errorReporter.emitError(stmt->getSourceRange(),errorDescription);
    return false;
  }
  const Attr* attr = expectedAttr.get();
  //INFO: no OKL attributes to process, continue
  if(!attr) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseStmt(stmt);
  }
  return attrManager.handleAttr(attr, subStmt, _session);
}
}
