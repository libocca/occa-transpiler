#include "core/ast_traversal/preorder_traversal_nlr.h"
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>

namespace {
using namespace oklt;
using namespace clang;

template <typename TraversalType, typename ExprType>
bool dispatchTraverseFunc(TraversalType& traversal, ExprType expr) {
    using PureType = std::remove_pointer_t<ExprType>;
    if constexpr (std::is_same_v<PureType, Stmt>) {
        auto* expr_ = [](auto* expr) {
            if (expr->getStmtClass() == clang::Stmt::AttributedStmtClass) {
                return cast<AttributedStmt>(expr)->getSubStmt();
            }
            return expr;
        }(expr);
        return traversal.RecursiveASTVisitor<PreorderNlrTraversal>::TraverseStmt(expr_);
    } else if constexpr (std::is_same_v<PureType, Decl>) {
        return traversal.RecursiveASTVisitor<PreorderNlrTraversal>::TraverseDecl(expr);
    } else if constexpr (std::is_same_v<PureType, TranslationUnitDecl>) {
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
        if (expr.hasAttrs()) {
            return getFirstAttr(expr.getAttrs());
        }
        return nullptr;
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
                  OklSemaCtx& sema) {
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

    return true;
}
}  // namespace
namespace oklt {

PreorderNlrTraversal::PreorderNlrTraversal(AstProcessorManager& procMng, SessionStage& stage)
    : _procMng(procMng),
      _stage(stage),
      _sema(_stage.tryEmplaceUserCtx<OklSemaCtx>()) {}

bool PreorderNlrTraversal::TraverseDecl(clang::Decl* decl) {
    return traverseExpr(*this, decl, _procMng, _stage, _sema);
}

bool PreorderNlrTraversal::TraverseStmt(clang::Stmt* stmt) {
    return traverseExpr(*this, stmt, _procMng, _stage, _sema);
}

bool PreorderNlrTraversal::TraverseTranslationUnitDecl(
    clang::TranslationUnitDecl* translationUnitDecl) {
    return traverseExpr(*this, translationUnitDecl, _procMng, _stage, _sema);
}

tl::expected<std::string, Error> PreorderNlrTraversal::applyAstProccessor(
    clang::TranslationUnitDecl* translationUnitDecl) {
    if (!TraverseTranslationUnitDecl(translationUnitDecl)) {
        return tl::make_unexpected(Error{{}, "AST trasrse failed"});
    }

    return _stage.getRewriterResultOfMainFile();
}
}  // namespace oklt
