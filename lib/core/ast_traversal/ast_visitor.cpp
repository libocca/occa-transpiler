#include "oklt/core/ast_traversal/ast_visitor.h"
#include "oklt/core/attribute_manager/attribute_manager.h"
#include "oklt/core/attribute_manager/attributed_type_map.h"
#include "oklt/core/transpiler_session/session_stage.h"
#include "oklt/core/transpiler_session/transpiler_session.h"

namespace oklt {
using namespace clang;

ASTVisitor::ASTVisitor(SessionStage& session) : _stage(session) {}

bool ASTVisitor::TraverseDecl(Decl* decl) {
  if (!decl->hasAttrs()) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseDecl(decl);
  }

  auto& attrManager = _stage.getAttrManager();
  auto expectedAttr = attrManager.checkAttrs(decl->getAttrs(), decl, _stage);
  if (!expectedAttr) {
    // TODO report diagnostic error using clang tooling
    //  auto &errorReporter = _session.getErrorReporter();
    //  auto errorDescription = expectedAttr.error().message();
    //  errorReporter.emitError(funcDecl->getSourceRange(),errorDescription);
    return false;
  }

  const Attr* attr = expectedAttr.value();
  // INFO: no OKL attributes to process, continue
  if (!attr) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseDecl(decl);
  }

  if (!attrManager.handleAttr(attr, decl, _stage)) {
    return false;
  }

  return RecursiveASTVisitor<ASTVisitor>::TraverseDecl(decl);
}

bool ASTVisitor::TraverseStmt(Stmt* stmt, DataRecursionQueue* queue) {
  if (!stmt) {
    return true;
  }
  if (stmt->getStmtClass() != Stmt::AttributedStmtClass) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseStmt(stmt, queue);
  }

  auto* attrStmt = cast<AttributedStmt>(stmt);
  auto& attrManager = _stage.getAttrManager();
  auto expectedAttr = attrManager.checkAttrs(attrStmt->getAttrs(), stmt, _stage);
  if (!expectedAttr) {
    // TODO report diagnostic error using clang tooling
    //  auto &errorReporter = _session.getErrorReporter();
    //  auto errorDescription = expectedAttr.error().message();
    //  errorReporter.emitError(stmt->getSourceRange(),errorDescription);
    return false;
  }

  const Attr* attr = expectedAttr.value();
  // INFO: no OKL attributes to process, continue
  if (!attr) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseStmt(stmt, queue);
  }

  const Stmt* subStmt = attrStmt->getSubStmt();
  if (!attrManager.handleAttr(attr, subStmt, _stage)) {
    return false;
  }

  return RecursiveASTVisitor<ASTVisitor>::TraverseStmt(stmt, queue);
}

bool ASTVisitor::TraverseRecoveryExpr(RecoveryExpr* expr, DataRecursionQueue* queue) {
  auto subExpr = expr->subExpressions();
  if (subExpr.empty()) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseRecoveryExpr(expr, queue);
  }

  auto declRefExpr = dyn_cast<DeclRefExpr>(subExpr[0]);
  if (!declRefExpr) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseRecoveryExpr(expr, queue);
  }

  auto& ctx = _stage.getCompiler().getASTContext();
  auto& attrTypeMap = _stage.tryEmplaceUserCtx<AttributedTypeMap>();
  auto attrs = attrTypeMap.get(ctx, declRefExpr->getType());

  auto& attrManager = _stage.getAttrManager();
  auto expectedAttr = attrManager.checkAttrs(attrs, expr, _stage);
  if (!expectedAttr) {
    // TODO report diagnostic error using clang tooling
    return false;
  }

  const Attr* attr = expectedAttr.value();
  if (!attrManager.handleAttr(attr, expr, _stage)) {
    return false;
  }

  return RecursiveASTVisitor<ASTVisitor>::TraverseRecoveryExpr(expr, queue);
}

}  // namespace oklt
