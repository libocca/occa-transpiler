#include "core/ast_traversal/preorder_traversal_nlr.h"
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace {
using namespace oklt;
using namespace clang;

template <typename TraversalType, typename ExprType>
bool dispatchTraverseFunc(TraversalType& traversal, ExprType expr) {
    if constexpr (std::is_same_v<ExprType, Stmt*>) {
        auto* expr_ = [](auto* expr) {
            if (expr->getStmtClass() == clang::Stmt::AttributedStmtClass) {
                return cast<AttributedStmt>(expr)->getSubStmt();
            }
            return expr;
        }(expr);
        return traversal.RecursiveASTVisitor<PreorderNlrTraversal>::TraverseStmt(expr_);
    } else if constexpr (std::is_same_v<ExprType, Decl*>) {
        return traversal.RecursiveASTVisitor<PreorderNlrTraversal>::TraverseDecl(expr);
    } else if constexpr (std::is_same_v<ExprType, RecoveryExpr*>) {
        return traversal.RecursiveASTVisitor<PreorderNlrTraversal>::TraverseRecoveryExpr(expr);
    } else if constexpr (std::is_same_v<ExprType, TranslationUnitDecl*>) {
        return traversal.RecursiveASTVisitor<PreorderNlrTraversal>::TraverseTranslationUnitDecl(
            expr);
    }
    return false;
}

template <typename TraversalType, typename ExprType>
bool traverseExpr(TraversalType& traversal,
                  ExprType expr,
                  AstProcessorManager& procMng,
                  SessionStage& stage) {
    if (expr == nullptr) {
        return true;
    }
    auto procType = stage.getAstProccesorType();
    if (!procMng.runPreActionNodeHandle(procType, expr, stage)) {
        return false;
    }

    // dispatch the next node
    if (!dispatchTraverseFunc(traversal, expr)) {
        return false;
    }
    return procMng.runPostActionNodeHandle(procType, expr, stage);
}
}  // namespace
namespace oklt {

PreorderNlrTraversal::PreorderNlrTraversal(AstProcessorManager& procMng, SessionStage& stage)
    : _procMng(procMng),
      _stage(stage) {}
bool PreorderNlrTraversal::TraverseDecl(clang::Decl* decl) {
    return traverseExpr(*this, decl, _procMng, _stage);
}

bool PreorderNlrTraversal::TraverseStmt(clang::Stmt* stmt) {
    return traverseExpr(*this, stmt, _procMng, _stage);
}

bool PreorderNlrTraversal::TraverseRecoveryExpr(clang::RecoveryExpr* recoveryExpr) {
    return traverseExpr(*this, recoveryExpr, _procMng, _stage);
}

bool PreorderNlrTraversal::TraverseTranslationUnitDecl(
    clang::TranslationUnitDecl* translationUnitDecl) {
    return traverseExpr(*this, translationUnitDecl, _procMng, _stage);
}

}  // namespace oklt
