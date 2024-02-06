#include <oklt/core/ast_traversal/semantic_mock.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include "oklt/core/attribute_manager/attributed_type_map.h"
#include <oklt/core/transpiler_session/session_stage.h>
#include "visit_overload.hpp"


namespace oklt {
using namespace clang;

SemanticMockVisitor::SemanticMockVisitor(SessionStage& stage)
    :SemanticASTVisitorBase(stage)
{}

bool SemanticMockVisitor::traverseTranslationUnit(clang::Decl* decl)
{
  return TraverseDecl(decl);
}

bool SemanticMockVisitor::TraverseDecl(clang::Decl* decl)
{
  if (!decl->hasAttrs()) {
    return RecursiveASTVisitor<SemanticMockVisitor>::TraverseDecl(decl);
  }
  auto attrs = decl->getAttrs();

  auto validationResult = SemanticASTVisitorBase::validateAttribute(attrs);
  auto validationHandlers = Overload {
    [this, decl](const clang::Attr*attr) -> bool {
      auto& attrManager = _stage.getAttrManager();
      if (!attrManager.handleAttr(attr, decl, _stage, nullptr)) {
        return false;
      }
      return RecursiveASTVisitor<SemanticMockVisitor>::TraverseDecl(decl);
    },
    [this, decl](const NoOKLAttrs&) -> bool{
      return RecursiveASTVisitor<SemanticMockVisitor>::TraverseDecl(decl);
    },
    [](const ErrorFired&) -> bool {
      return false;
    },
  };
  return std::visit(validationHandlers, validationResult);
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

  auto validationResult = SemanticASTVisitorBase::validateAttribute(attrs);
  auto validationHandlers = Overload {
    [this, attrStmt, queue, stmt](const clang::Attr*attr) -> bool {
      auto& attrManager = _stage.getAttrManager();
      const Stmt* subStmt = attrStmt->getSubStmt();
      if (!attrManager.handleAttr(attr, subStmt, _stage, nullptr)) {
        return false;
      }
      return RecursiveASTVisitor<SemanticMockVisitor>::TraverseStmt(stmt, queue);;
    },
    [this, stmt, queue](const NoOKLAttrs&) -> bool{
      return RecursiveASTVisitor<SemanticMockVisitor>::TraverseStmt(stmt, queue);
    },
    [](const ErrorFired&) -> bool {
      return false;
    },
  };
  return std::visit(validationHandlers, validationResult);
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

  auto validationResult = SemanticASTVisitorBase::validateAttribute(attrs);
  auto validationHandlers = Overload {
    [this, expr, queue](const clang::Attr* attr) -> bool {
      auto& attrManager = _stage.getAttrManager();
      if (!attrManager.handleAttr(attr, expr, _stage, nullptr)) {
        return false;
      }
      return RecursiveASTVisitor<SemanticMockVisitor>::TraverseRecoveryExpr(expr, queue);;
    },
    [this, expr, queue](const NoOKLAttrs&) -> bool{
      return RecursiveASTVisitor<SemanticMockVisitor>::TraverseRecoveryExpr(expr, queue);
    },
    [](const ErrorFired&) -> bool {
      return false;
    },
  };
  return std::visit(validationHandlers, validationResult);
}

}  // namespace oklt
