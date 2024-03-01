#include <oklt/util/string_utils.h>
#include "attributes/attribute_names.h"
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/ast_processors/okl_sema_processor/handlers/function.h"
#include "core/ast_processors/okl_sema_processor/handlers/loop.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/attribute_manager/attributed_type_map.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

namespace {
using namespace clang;
using namespace oklt;

const Attr* getOklAttrImpl(tl::expected<const Attr*, Error> attrResult,
                           SessionStage& stage,
                           std::string_view attrName = {}) {
    if (!attrResult) {
        stage.pushError(std::move(attrResult.error()));
        return nullptr;
    }

    const auto* attr = attrResult.value();
    if (!attr) {
        return nullptr;
    }

    if (attrName.empty()) {
        return attr;
    }

    if (attr->getNormalizedFullName() != attrName) {
        return nullptr;
    }

    return attr;
}

const Attr* getOklAttr(const Decl& decl, SessionStage& stage, std::string_view attrName = {}) {
    if (decl.hasAttrs()) {
        auto attrResult = stage.getAttrManager().checkAttrs(decl.getAttrs(), decl, stage);
        return getOklAttrImpl(attrResult, stage, attrName);
    }
    return nullptr;
}

const Attr* getOklAttr(const AttributedStmt& stmt,
                       SessionStage& stage,
                       std::string_view attrName = {}) {
    auto attrResult = stage.getAttrManager().checkAttrs(stmt.getAttrs(), stmt, stage);
    return getOklAttrImpl(attrResult, stage, attrName);
}

bool dispatchPreValidationAttrStmtSema(const Attr& attr,
                                       const Stmt& stmt,
                                       SessionStage& stage,
                                       OklSemaCtx& sema) {
#ifdef OKL_SEMA_DEBUG_LOG
    llvm::outs() << "stmt: " << stmt.getStmtClassName() << '\n'
                 << "attr: " << attr.getNormalizedFullName() << '\n';
#endif
    switch (stmt.getStmtClass()) {
        case Stmt::ForStmtClass:
            return preValidateOklForLoopSema(attr, cast<ForStmt>(stmt), stage, sema);
        default:
            break;
    }
    return true;
}

bool dispatchPostValidationAttrStmtSema(const Attr& attr,
                                        const Stmt& stmt,
                                        SessionStage& stage,
                                        OklSemaCtx& sema) {
    switch (stmt.getStmtClass()) {
        case Stmt::ForStmtClass:
            return postValidateOklForLoopSema(attr, cast<ForStmt>(stmt), stage, sema);
        default:
            break;
    }
    return true;
}

template <typename ExprType>
HandleResult runExprTranspilerHanders(const ExprType& expr,
                                      SessionStage& stage,
                                      std::string_view attrName = {}) {
    auto* attr = getOklAttr(expr, stage, attrName);
    if (!attr) {
        if constexpr (std::is_same_v<ExprType, Stmt>) {
            return stage.getAttrManager().handleStmt(expr, stage);
        } else if constexpr (std::is_same_v<ExprType, AttributedStmt>) {
            return stage.getAttrManager().handleStmt(*expr.getSubStmt(), stage);
        } else {
            return stage.getAttrManager().handleDecl(expr, stage);
        }
        return {};
    }

    // finally parser it
    auto& am = stage.getAttrManager();
    auto params = am.parseAttr(*attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    // run specific kernel attribute handler
    if constexpr (std::is_same_v<ExprType, AttributedStmt>) {
        // Get statement from attributed statement
        auto* subStmt = expr.getSubStmt();
        if (!subStmt) {
            return tl::make_unexpected(Error());
        }
        return am.handleAttr(*attr, *subStmt, &params.value(), stage);
    } else {
        return am.handleAttr(*attr, expr, &params.value(), stage);
    }

    return {};
}
///////////////////// handlers entry points ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// Generic Decl pre handlers
HandleResult runPreActionDecl(const Decl& decl, SessionStage& stage) {
    return {};
}

// Generic Decl post handlers
HandleResult runPostActionDecl(const Decl& decl, SessionStage& stage) {
    return runExprTranspilerHanders(decl, stage);
}

// OKL kernel sema validator
HandleResult validateFunctionDecl(const FunctionDecl& fd, SessionStage& stage) {
    // we interesting only in OKL kernel function
    auto attr = getOklAttr(fd, stage, KERNEL_ATTR_NAME);
    if (!attr) {
        return {};
    }

    // go though sema validation
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!preValidateOklKernelSema(fd, stage, sema)) {
        return tl::make_unexpected(Error());
    }

    return {};
}

HandleResult transpileFunctionDecl(const FunctionDecl& fd, SessionStage& stage) {
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();

    // ensure it is backward path for current parsing OKL kernel
    if (!sema.isCurrentParsingOklKernel(fd)) {
        return stage.getAttrManager().handleDecl(fd, stage);
    }

    auto result = runExprTranspilerHanders(fd, stage, KERNEL_ATTR_NAME);
    if (!result) {
        return result;
    }

    // finalize sema tranpsilation of OKL kernel
    if (!postValidateOklKernelSema(fd, stage, sema)) {
        return tl::make_unexpected(Error());
    }

    return result;
}

// OKL kernel parameters sema validator
HandleResult validateParmDecl(const ParmVarDecl& parm, SessionStage& stage) {
    // not inside OKL kernel
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.isParsingOklKernel()) {
        return {};
    }

    // run param sema
    if (!preValidateOklKernelParamSema(parm, stage, sema)) {
        return tl::make_unexpected(Error());
    }

    return {};
}

HandleResult transpileParmDecl(const ParmVarDecl& parm, SessionStage& stage) {
    // for attributed parm decl backend handler should set arg raw string representation
    auto* attr = getOklAttr(parm, stage);
    if (attr) {
        // or parse it if attributed
        auto params = stage.getAttrManager().parseAttr(*attr, stage);
        if (!params) {
            return tl::make_unexpected(params.error());
        }
        return stage.getAttrManager().handleAttr(*attr, parm, &params.value(), stage);
    } else {
        // for regular parm decl sema sets raw string representation
        auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
        if (!postValidateOklKernelParamSema(parm, stage, sema)) {
            return tl::make_unexpected(Error());
        }
    }

    return {};
}

HandleResult runPreActionAttrStmt(const AttributedStmt& attrStmt, SessionStage& stage) {
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.isParsingOklKernel()) {
        //  make approptiate error code
        return tl::make_unexpected(
            Error{std::error_code(), "OKL attribute inside of non OKL kernel function"});
    }

    auto* attr = getOklAttr(attrStmt, stage);
    if (!attr) {
        auto attrName = attrStmt.getAttrs()[0]->getNormalizedFullName();
        return tl::make_unexpected(
            Error{std::error_code(),
                  util::fmt("No backend ({}) handler registered for attribute {}",
                            backendToString(stage.getBackend()),
                            attrName)
                      .value()});
    }

    // dispatch specific sema handler
    if (!dispatchPreValidationAttrStmtSema(*attr, *attrStmt.getSubStmt(), stage, sema)) {
        return tl::make_unexpected(Error());
    }

    return {};
}

HandleResult runPostActionAttrStmt(const AttributedStmt& attrStmt, SessionStage& stage) {
    // legacy OKL applies one attribute per stmt/decl
    auto result = runExprTranspilerHanders(attrStmt, stage);
    if (!result) {
        return result;
    }

    auto* attr = getOklAttr(attrStmt, stage);
    if (!attr) {
        return tl::make_unexpected(Error());
    }

    // sema transpiler action
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!dispatchPostValidationAttrStmtSema(*attr, *attrStmt.getSubStmt(), stage, sema)) {
        return tl::make_unexpected(Error());
    }

    return result;
}

HandleResult runPreActionRecoveryExpr(const RecoveryExpr& expr, SessionStage& stage) {
    return {};
}

HandleResult runPostActionRecoveryExpr(const RecoveryExpr& expr, SessionStage& stage) {
    auto subExpr = expr.subExpressions();
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
    auto expectedAttr = am.checkAttrs(attrs, expr, stage);
    if (!expectedAttr) {
        // TODO report diagnostic error using clang tooling
        return {};
    }

    const Attr* attr = expectedAttr.value();
    return am.handleAttr(*attr, expr, {}, stage);
}

__attribute__((constructor)) void registerAstNodeHanlder() {
    auto& mng = AstProcessorManager::instance();
    using DeclHandle = AstProcessorManager::DeclNodeHandle;
    using StmtHandle = AstProcessorManager::StmtNodeHandle;

    // generic decl except specific handlers
    auto ok = mng.registerGenericHandle(
        AstProcessorType::OKL_WITH_SEMA,
        DeclHandle{.preAction = runPreActionDecl, .postAction = runPostActionDecl});
    assert(ok);

    // exam for OKL kernel
    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, Decl::Function},
        makeSpecificDeclHandle(validateFunctionDecl, transpileFunctionDecl));
    assert(ok);

    // exam for OKL kernel parametets
    ok =
        mng.registerSpecificNodeHandle({AstProcessorType::OKL_WITH_SEMA, Decl::ParmVar},
                                       makeSpecificDeclHandle(validateParmDecl, transpileParmDecl));
    assert(ok);

    // all OKL attributed stmt
    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, Stmt::AttributedStmtClass},
        makeSpecificStmtHandle(runPreActionAttrStmt, runPostActionAttrStmt));
    assert(ok);

    // recovery expr for handling dim
    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, Stmt::RecoveryExprClass},
        makeSpecificStmtHandle(runPreActionRecoveryExpr, runPostActionRecoveryExpr));
    assert(ok);
}
}  // namespace
