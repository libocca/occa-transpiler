#include <oklt/util/string_utils.h>
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/attribute_manager/attributed_type_map.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

namespace {
using namespace clang;
using namespace oklt;

HandleResult runPreActionDecl(const Decl& decl, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " decl name: " << decl.getDeclKindName() << '\n';
#endif
    return {};
}

HandleResult runPostActionDecl(const clang::Decl& decl, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " decl name: " << decl.getDeclKindName() << '\n';
#endif

    auto& am = stage.getAttrManager();
    if (!decl.hasAttrs()) {
        return am.handleDecl(decl, stage);
    }

    auto expectedAttr = am.checkAttrs(decl.getAttrs(), decl, stage);
    if (!expectedAttr) {
        // TODO report diagnostic error using clang tooling
        //  auto &errorReporter = _session.getErrorReporter();
        //  auto errorDescription = expectedAttr.error().message();
        //  errorReporter.emitError(funcDecl->getSourceRange(),errorDescription);
        return {};
    }

    const Attr* attr = expectedAttr.value();
    if (!attr) {
        return {};
    }

    auto params = am.parseAttr(*attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    return am.handleAttr(*attr, decl, &params.value(), stage);
}

HandleResult runPreActionStmt(const clang::Stmt& stmt, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << stmt.getStmtClassName() << '\n';
#endif
    return {};
}

HandleResult runPostActionStmt(const clang::Stmt& stmt, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << stmt.getStmtClassName() << '\n';
#endif
    return {};
}

HandleResult runPreActionAttrStmt(const clang::AttributedStmt& attrStmt, SessionStage& stage) {
    return {};
}

HandleResult runPostActionAttrStmt(const clang::AttributedStmt& attrStmt, SessionStage& stage) {
    auto& am = stage.getAttrManager();
    auto expectedAttr = am.checkAttrs(attrStmt.getAttrs(), attrStmt, stage);
    if (!expectedAttr) {
        // TODO report diagnostic error using clang tooling
        //  auto &errorReporter = _session.getErrorReporter();
        //  auto errorDescription = expectedAttr.error().message();
        //  errorReporter.emitError(stmt->getSourceRange(),errorDescription);
        return {};
    }

    const Attr* attr = expectedAttr.value();
    // INFO: no OKL attributes to process, continue
    if (!attr) {
        return {};
    }

    auto params = am.parseAttr(*attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    if (!params) {
        return tl::make_unexpected(Error());
    }

    const Stmt* subStmt = attrStmt.getSubStmt();
    return am.handleAttr(*attr, *subStmt, &params.value(), stage);
}

HandleResult runPreActionRecoveryExpr(const clang::RecoveryExpr& expr, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << expr.getStmtClassName() << '\n';
#endif
    return {};
}

HandleResult runPostActionRecoveryExpr(const clang::RecoveryExpr& expr_, SessionStage& stage) {
    auto* expr = dyn_cast_or_null<RecoveryExpr>(&expr_);
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << expr->getStmtClassName() << '\n';
#endif
    auto subExpr = expr->subExpressions();
    if (subExpr.empty()) {
        return {};
    }

    auto declRefExpr = dyn_cast<DeclRefExpr>(subExpr[0]);
    if (!declRefExpr) {
        return {};
    }

    auto& ctx = stage.getCompiler().getASTContext();
    auto& attrTypeMap = stage.tryEmplaceUserCtx<AttributedTypeMap>();
    auto attrs = attrTypeMap.get(ctx, declRefExpr->getType());

    auto& am = stage.getAttrManager();
    auto placeholder = am.checkAttrs(attrs, *expr, stage);
    auto expectedAttr = placeholder;
    if (!expectedAttr) {
        // TODO report diagnostic error using clang tooling
        return {};
    }

    const Attr* attr = expectedAttr.value();
    auto* params = stage.getUserCtx(util::pointerToStr(attr));
    return am.handleAttr(*attr, *expr, params, stage);
}

__attribute__((constructor)) void registerAstNodeHanlder() {
    auto& mng = AstProcessorManager::instance();
    using DeclHandle = AstProcessorManager::DeclNodeHandle;
    using StmtHandle = AstProcessorManager::StmtNodeHandle;

    auto ok = mng.registerGenericHandle(
        AstProcessorType::OKL_NO_SEMA,
        DeclHandle{.preAction = runPreActionDecl, .postAction = runPostActionDecl});
    assert(ok);

    ok = mng.registerGenericHandle(
        AstProcessorType::OKL_NO_SEMA,
        StmtHandle{.preAction = runPreActionStmt, .postAction = runPostActionStmt});
    assert(ok);

    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_NO_SEMA, Stmt::AttributedStmtClass},
        makeSpecificStmtHandle(runPreActionAttrStmt, runPostActionAttrStmt));
    assert(ok);

    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_NO_SEMA, Stmt::RecoveryExprClass},
        makeSpecificStmtHandle(runPreActionRecoveryExpr, runPostActionRecoveryExpr));
    assert(ok);
}
}  // namespace
