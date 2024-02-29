#include "core/ast_traversal/preorder_traversal_nlr.h"
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpilation.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>

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

template <typename ArrayType>
const Attr* getFirstAttr(const ArrayType& attrs) {
    if (attrs.empty()) {
        return nullptr;
    }
    return attrs[0];
}

template <typename ExprType>
const Attr* tryGetAttr(ExprType& expr) {
    if constexpr (std::is_base_of_v<Decl, ExprType>) {
        return getFirstAttr(expr.getAttrs());
    }
    if constexpr (std::is_base_of_v<Stmt, ExprType>) {
        if (expr.getStmtClass() != Stmt::AttributedStmtClass) {
            return nullptr;
        }
        return getFirstAttr(cast<AttributedStmt>(expr).getAttrs());
    }

    return nullptr;
}

const Decl& tryGetAttrExpr(const Decl& d) {
    return d;
}

const Stmt& tryGetAttrExpr(const Stmt& s) {
    if (s.getStmtClass() != Stmt::AttributedStmtClass) {
        return s;
    }
    return *cast<AttributedStmt>(s).getSubStmt();
}

template <typename TraversalType, typename ExprType>
bool traverseExpr(TraversalType& traversal,
                  ExprType* expr,
                  AstProcessorManager& procMng,
                  SessionStage& stage,
                  OklSemaCtx& sema,
                  Transpilations& transpilations) {
    if (expr == nullptr) {
        return true;
    }

    auto procType = stage.getAstProccesorType();
    const auto* attr = tryGetAttr(*expr);
    const auto& expr_ = tryGetAttrExpr(*expr);

    auto result = procMng.runPreActionNodeHandle(procType, attr, expr_, sema, stage);
    if (!result) {
        stage.pushError(std::move(result.error()));
        return false;
    }

    if (!isEmpty(result.value())) {
        transpilations.emplace_back(std::move(result.value()));
    }

    // dispatch the next node
    if (!dispatchTraverseFunc(traversal, expr)) {
        stage.pushError(Error{.ec = std::error_code(), .desc = "trasverse is stopped"});
        return false;
    }

    result = procMng.runPostActionNodeHandle(procType, attr, expr_, sema, stage);
    if (!result) {
        stage.pushError(std::move(result.error()));
        return false;
    }

    if (!isEmpty(result.value())) {
        transpilations.emplace_back(std::move(result.value()));
    }

    return true;
}
}  // namespace
namespace oklt {

PreorderNlrTraversal::PreorderNlrTraversal(AstProcessorManager& procMng, SessionStage& stage)
    : _procMng(procMng),
      _stage(stage),
      _sema(_stage.tryEmplaceUserCtx<OklSemaCtx>()) {}

bool PreorderNlrTraversal::TraverseDecl(clang::Decl* decl) {
    return traverseExpr(*this, decl, _procMng, _stage, _sema, _trasnpilations);
}

bool PreorderNlrTraversal::TraverseStmt(clang::Stmt* stmt) {
    return traverseExpr(*this, stmt, _procMng, _stage, _sema, _trasnpilations);
}

bool PreorderNlrTraversal::TraverseRecoveryExpr(clang::RecoveryExpr* recoveryExpr) {
    return traverseExpr(*this, recoveryExpr, _procMng, _stage, _sema, _trasnpilations);
}

bool PreorderNlrTraversal::TraverseTranslationUnitDecl(
    clang::TranslationUnitDecl* translationUnitDecl) {
    return traverseExpr(*this, translationUnitDecl, _procMng, _stage, _sema, _trasnpilations);
}

tl::expected<std::string, std::error_code> PreorderNlrTraversal::applyAstProccessor(
    clang::TranslationUnitDecl* translationUnitDecl) {
    if (!TraverseTranslationUnitDecl(translationUnitDecl)) {
        return tl::make_unexpected(std::error_code());
    }

    auto ok = applyTranspilations(_trasnpilations, _stage.getRewriter());
    if (!ok) {
        return tl::make_unexpected(std::error_code());
    }

    return _stage.getRewriterResult();
}
}  // namespace oklt
