#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/attribute_manager/attributed_type_map.h"
#include "core/transpiler_session/session_stage.h"
#include <oklt/util/string_utils.h>

#include <clang/AST/AST.h>

namespace {
using namespace clang;
using namespace oklt;

bool runPreActionDecl(const Decl& decl, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " decl name: " << decl.getDeclKindName() << '\n';
#endif
    auto& am = stage.getAttrManager();
    if (!decl.hasAttrs()) {
        return true;
    }

    for (const auto attr : decl.getAttrs()) {
        if (!attr)
            continue;

        auto params = am.parseAttr(*attr, stage);
        if (!params) {
            stage.pushError(params.error());
            return false;
        }
        stage.setUserCtx(util::pointerToStr(attr), params.value());
    }
    return true;
}

bool runPostActionDecl(const Decl& decl, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " decl name: " << decl.getDeclKindName() << '\n';
#endif

    auto& am = stage.getAttrManager();
    if (!decl.hasAttrs()) {
        auto ok = am.handleDecl(decl, stage);
        if (!ok) {
            stage.pushError(ok.error());
            return false;
        }
        return true;
    }

    auto expectedAttr = am.checkAttrs(decl.getAttrs(), decl, stage);
    if (!expectedAttr) {
        // TODO report diagnostic error using clang tooling
        //  auto &errorReporter = _session.getErrorReporter();
        //  auto errorDescription = expectedAttr.error().message();
        //  errorReporter.emitError(funcDecl->getSourceRange(),errorDescription);
        return true;
    }

    const Attr* attr = expectedAttr.value();
    if (!attr) {
        return true;
    }

    auto params = stage.getUserCtx(util::pointerToStr(attr));
    if (!params) {
        return false;
    }
    auto ok = am.handleAttr(*attr, decl, params, stage);
    if (!ok) {
        stage.pushError(ok.error());
        return false;
    }
    return true;
}

bool runPreActionStmt(const Stmt& stmt, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << stmt.getStmtClassName() << '\n';
#endif
    return true;
}

bool runPostActionStmt(const Stmt& stmt, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << stmt.getStmtClassName() << '\n';
#endif
    return true;
}

bool runPreActionAttrStmt(const AttributedStmt& attrStmt, SessionStage& stage) {
    auto& am = stage.getAttrManager();
    for (const auto attr : attrStmt.getAttrs()) {
        if (!attr)
            continue;

        auto params = am.parseAttr(*attr, stage);
        if (!params) {
            stage.pushError(params.error());
            return false;
        }
        stage.setUserCtx(util::pointerToStr(attr), params.value());
    }

    return true;
}

bool runPostActionAttrStmt(const AttributedStmt& attrStmt, SessionStage& stage) {
    auto& am = stage.getAttrManager();
    auto expectedAttr = am.checkAttrs(attrStmt.getAttrs(), attrStmt, stage);
    if (!expectedAttr) {
        // TODO report diagnostic error using clang tooling
        //  auto &errorReporter = _session.getErrorReporter();
        //  auto errorDescription = expectedAttr.error().message();
        //  errorReporter.emitError(stmt->getSourceRange(),errorDescription);
        return true;
    }

    const Attr* attr = expectedAttr.value();
    // INFO: no OKL attributes to process, continue
    if (!attr) {
        return true;
    }

    auto params = stage.getUserCtx(util::pointerToStr(attr));
    if (!params) {
        return false;
    }
    auto ok = am.handleAttr(*attr, *attrStmt.getSubStmt(), params, stage);
    if (!ok) {
        stage.pushError(ok.error());
        return false;
    }

    return true;
}

bool runPreActionRecoveryExpr(const RecoveryExpr& expr, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << expr.getStmtClassName() << '\n';
#endif
    return true;
}

bool runPostActionRecoveryExpr(const RecoveryExpr& expr, SessionStage& stage) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << __PRETTY_FUNCTION__ << " stmt name: " << expr.getStmtClassName() << '\n';
#endif
    auto subExpr = expr.subExpressions();
    if (subExpr.empty()) {
        return true;
    }

    auto declRefExpr = dyn_cast<DeclRefExpr>(subExpr[0]);
    if (!declRefExpr) {
        return true;
    }

    auto& ctx = stage.getCompiler().getASTContext();
    auto& attrTypeMap = stage.tryEmplaceUserCtx<AttributedTypeMap>();
    auto attrs = attrTypeMap.get(ctx, declRefExpr->getType());

    auto& am = stage.getAttrManager();
    auto expectedAttr = am.checkAttrs(attrs, expr, stage);
    if (!expectedAttr) {
        // TODO report diagnostic error using clang tooling
        return true;
    }

    const Attr* attr = expectedAttr.value();
    // INFO: no OKL attributes to process, continue
    if (!attr) {
        return true;
    }

    auto* params = stage.getUserCtx(util::pointerToStr(attr));
    auto ok = am.handleAttr(*attr, expr, params, stage);
    if (!ok) {
        stage.pushError(ok.error());
        return false;
    }
    return true;
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
