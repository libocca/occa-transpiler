#include <oklt/core/ast_traversal/semantic_mock.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include "oklt/core/attribute_manager/attributed_type_map.h"
#include <oklt/core/transpiler_session/session_stage.h>
#include <variant>
// #include "visit_overload.hpp"


namespace oklt {
using namespace clang;

SemanticMockVisitor::SemanticMockVisitor(SessionStage& stage,
                                         AttrValidatorFnType attrValidateFn)
    :_stage(stage)
    , _attrValidator(attrValidateFn)
{}

bool SemanticMockVisitor::TraverseDecl(clang::Decl* decl)
{
  if (!decl->hasAttrs()) {
    return RecursiveASTVisitor<SemanticMockVisitor>::TraverseDecl(decl);
  }
  auto attrs = decl->getAttrs();

  auto validationResult = _attrValidator(attrs, _stage);
  if(!validationResult) {
    return false;
  }
  auto maybeAttr = validationResult.value();
  if(!maybeAttr) {
    return RecursiveASTVisitor<SemanticMockVisitor>::TraverseDecl(decl);
  }
  auto& attrManager = _stage.getAttrManager();
  auto handlerResult = attrManager.handleAttr(maybeAttr, decl, _stage);
  if(!handlerResult) {
    return false;
  }
  return RecursiveASTVisitor<SemanticMockVisitor>::TraverseDecl(decl);
}

bool SemanticMockVisitor::TraverseStmt(clang::Stmt* stmt, DataRecursionQueue* queue)
{
  if (!stmt) {
    return true;
  }
  if (stmt->getStmtClass() != Stmt::AttributedStmtClass) {
    return RecursiveASTVisitor<SemanticMockVisitor>::TraverseStmt(stmt, queue);
  }

  auto* attrStmt = cast<AttributedStmt>(stmt);
  auto attrs = attrStmt->getAttrs();

  auto validationResult = _attrValidator(attrs, _stage);
  if(!validationResult) {
    return false;
  }
  auto maybeAttr = validationResult.value();
  if(!maybeAttr) {
    return RecursiveASTVisitor<SemanticMockVisitor>::TraverseStmt(stmt, queue);
  }
  auto& attrManager = _stage.getAttrManager();
  const Stmt* subStmt = attrStmt->getSubStmt();
  auto handleResult = attrManager.handleAttr(maybeAttr, subStmt, _stage);
  if (!handleResult) {
    return false;
  }
  return RecursiveASTVisitor<SemanticMockVisitor>::TraverseStmt(stmt, queue);
}

bool SemanticMockVisitor::TraverseRecoveryExpr(clang::RecoveryExpr* expr, DataRecursionQueue* queue)
{
  auto subExpr = expr->subExpressions();
  if (subExpr.empty()) {
    return RecursiveASTVisitor<SemanticMockVisitor>::TraverseRecoveryExpr(expr, queue);
  }

  auto declRefExpr = dyn_cast<DeclRefExpr>(subExpr[0]);
  if (!declRefExpr) {
    return RecursiveASTVisitor<SemanticMockVisitor>::TraverseRecoveryExpr(expr, queue);
  }

  auto& ctx = _stage.getCompiler().getASTContext();
  auto& attrTypeMap = _stage.tryEmplaceUserCtx<AttributedTypeMap>();
  auto attrs = attrTypeMap.get(ctx, declRefExpr->getType());

  auto validationResult = _attrValidator(attrs, _stage);
  if(!validationResult) {
    return false;
  }
  auto maybeAttr = validationResult.value();
  if(!maybeAttr) {
    return RecursiveASTVisitor<SemanticMockVisitor>::TraverseRecoveryExpr(expr, queue);
  }

  auto& attrManager = _stage.getAttrManager();
  auto handleResult = attrManager.handleAttr(maybeAttr, expr, _stage);
  if (!handleResult) {
    return false;
  }
  return RecursiveASTVisitor<SemanticMockVisitor>::TraverseRecoveryExpr(expr, queue);
}

}  // namespace oklt
