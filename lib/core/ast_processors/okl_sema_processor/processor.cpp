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

template <typename ExprType>
const clang::Attr* getOklAttr(const ExprType& expr,
                              SessionStage& stage,
                              std::string_view attrName = {}) {
    auto attrResult = stage.getAttrManager().checkAttrs(expr.getAttrs(), expr, stage);
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
bool runExprTranspilerHanders(const ExprType& expr,
                              SessionStage& stage,
                              std::string_view attrName = {},
                              bool continueIfNoAttrs = true) {
    auto* attr = getOklAttr(expr, stage, attrName);
    if (!attr) {
        if (continueIfNoAttrs) {
            if constexpr (std::is_same_v<ExprType, Stmt>) {
                return stage.getAttrManager().handleStmt(expr, stage).has_value();
            } else if constexpr (std::is_same_v<ExprType, AttributedStmt>) {
                return stage.getAttrManager().handleStmt(*expr.getSubStmt(), stage).has_value();
            } else {
                return stage.getAttrManager().handleDecl(expr, stage).has_value();
            }
        }
        return continueIfNoAttrs;
    }

    // finally parser it
    auto params = stage.getAttrManager().parseAttr(*attr, stage);
    if (!params) {
        stage.pushError(params.error());
        return false;
    }

    // run specific kernel attribute handler
    auto& am = stage.getAttrManager();
    if constexpr (std::is_same_v<ExprType, AttributedStmt>) {
        // Get statement from attributed statement
        auto ok = am.handleAttr(*attr, *expr.getSubStmt(), &params.value(), stage);
        if (!ok) {
            stage.pushError(ok.error());
            return false;
        }
    } else {
        auto ok = am.handleAttr(*attr, expr, &params.value(), stage);
        if (!ok) {
            stage.pushError(ok.error());
            return false;
        }
    }

    return true;
}
///////////////////// handlers entry points ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// Generic Decl pre handlers
bool runPreActionDecl(const Decl& decl, SessionStage& stage) {
    return true;
}

// Generic Decl post handlers
bool runPostActionDecl(const clang::Decl& decl, SessionStage& stage) {
    return runExprTranspilerHanders(decl, stage);
}

// OKL kernel sema validator
bool validateFunctionDecl(const FunctionDecl& fd, SessionStage& stage) {
    // we interested only in OKL kernel function
    auto attr = getOklAttr(fd, stage, KERNEL_ATTR_NAME);
    if (!attr) {
        return true;
    }

    // go through sema validation
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!preValidateOklKernelSema(fd, stage, sema)) {
        return false;
    }

    return true;
}

bool transpileFunctionDecl(const FunctionDecl& fd, SessionStage& stage) {
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();

    // ensure it is backward path for current parsing OKL kernel
    if (!sema.isCurrentParsingOklKernel(fd)) {
        return stage.getAttrManager().handleDecl(fd, stage).has_value();
    }

    if (!runExprTranspilerHanders(fd, stage, KERNEL_ATTR_NAME, false)) {
        return false;
    }

    // finalize sema tranpsilation of OKL kernel
    if (!postValidateOklKernelSema(fd, stage, sema)) {
        return false;
    }

    return true;
}

// OKL kernel parameters sema validator
bool validateParmDecl(const ParmVarDecl& parm, SessionStage& stage) {
    // not inside OKL kernel
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.isParsingOklKernel()) {
        return true;
    }

    // run param sema
    if (!preValidateOklKernelParamSema(parm, stage, sema)) {
        return false;
    }

    return true;
}

bool transpileParmDecl(const ParmVarDecl& parm, SessionStage& stage) {
    // for attributed parm decl backend hadnler should set arg raw string representation
    const auto attr = getOklAttr(parm, stage);
    if (attr) {
        // or parse it if attributed
        auto params = stage.getAttrManager().parseAttr(*attr, stage);
        if (!params) {
            stage.pushError(params.error());
            return false;
        }
        auto handleResult = stage.getAttrManager().handleAttr(*attr, parm, &params.value(), stage);
        if (!handleResult) {
            stage.pushError(handleResult.error());
            return false;
        }
    } else {
        // for regular parm decl sema sets raw string representation
        auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
        if (!postValidateOklKernelParamSema(parm, stage, sema)) {
            return false;
        }
    }

    return true;
}

bool runPreActionAttrStmt(const clang::AttributedStmt& attrStmt, SessionStage& stage) {
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.isParsingOklKernel()) {
        //  make approptiate error code
        stage.pushError(std::error_code(), "OKL attribute inside of non OKL kernel function");
        return false;
    }

    auto* attr = getOklAttr(attrStmt, stage);
    if (!attr) {
        return false;
    }

    // dispatch specific sema handler
    if (!dispatchPreValidationAttrStmtSema(*attr, *attrStmt.getSubStmt(), stage, sema)) {
        return false;
    }

    return true;
}

bool runPostActionAttrStmt(const clang::AttributedStmt& attrStmt, SessionStage& stage) {
    // legacy OKL applies one attribute per stmt/decl
    if (!runExprTranspilerHanders(attrStmt, stage)) {
        return false;
    }

    auto* attr = getOklAttr(attrStmt, stage);
    if (!attr) {
        return false;
    }

    // sema transpiler action
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!dispatchPostValidationAttrStmtSema(*attr, *attrStmt.getSubStmt(), stage, sema)) {
        return false;
    }

    return true;
}

bool runPreActionRecoveryExpr(const clang::RecoveryExpr& expr, SessionStage& stage) {
    return true;
}

bool runPostActionRecoveryExpr(const clang::RecoveryExpr& expr, SessionStage& stage) {
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
    return am.handleAttr(*attr, expr, params, stage).has_value();
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
