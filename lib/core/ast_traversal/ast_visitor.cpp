#include "oklt/core/ast_traversal/ast_visitor.h"
#include "oklt/core/transpiler_session/session_stage.h"
#include "oklt/core/attribute_manager/attribute_manager.h"
#include "oklt/core/attribute_manager/attributed_type_map.h"
#include "oklt/core/transpiler_session/transpiler_session.h"

namespace oklt {
using namespace clang;

ASTVisitor::ASTVisitor(SessionStage& session) : _session(session) {}

bool ASTVisitor::TraverseDecl(Decl* decl) {
  if (!decl->hasAttrs()) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseDecl(decl);
  }

  auto& attrManager = _session.getAttrManager();
  llvm::Expected<const Attr*> expectedAttr =
    attrManager.checkAttrs(decl->getAttrs(), decl, _session);
  if (!expectedAttr) {
    // auto &errorReporter = _session.getErrorReporter();
    auto errorDescription = toString(expectedAttr.takeError());
    // errorReporter.emitError(funcDecl->getSourceRange(),errorDescription);
    return false;
  }

  const Attr* attr = expectedAttr.get();
  // INFO: no OKL attributes to process, continue
  if (!attr) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseDecl(decl);
  }

  if (!attrManager.handleAttr(attr, decl, _session)) {
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
  auto& attrManager = _session.getAttrManager();
  auto expectedAttr = attrManager.checkAttrs(attrStmt->getAttrs(), stmt, _session);
  if (!expectedAttr) {
    // auto &errorReporter = _session.getErrorReporter();
    auto errorDescription = toString(expectedAttr.takeError());
    // errorReporter.emitError(stmt->getSourceRange(),errorDescription);
    return false;
  }

  const Attr* attr = expectedAttr.get();
  // INFO: no OKL attributes to process, continue
  if (!attr) {
    return RecursiveASTVisitor<ASTVisitor>::TraverseStmt(stmt, queue);
  }

  const Stmt* subStmt = attrStmt->getSubStmt();
  if (!attrManager.handleAttr(attr, subStmt, _session)) {
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

  auto& ctx = _session.getCompiler().getASTContext();
  auto& attrTypeMap = _session.tryEmplaceUserCtx<AttributedTypeMap>();
  auto attrs = attrTypeMap.get(ctx, declRefExpr->getType());

  auto& attrManager = _session.getAttrManager();
  llvm::Expected<const Attr*> expectedAttr = attrManager.checkAttrs(attrs, expr, _session);
  if (!expectedAttr) {
    return false;
  }

  const Attr* attr = expectedAttr.get();
  if (!attrManager.handleAttr(attr, expr, _session)) {
    return false;
  }

  return RecursiveASTVisitor<ASTVisitor>::TraverseRecoveryExpr(expr, queue);
}

}  // namespace oklt
