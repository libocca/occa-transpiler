#include <oklt/util/string_utils.h>
#include "attributes/attribute_names.h"
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/ast_processors/default_actions.h"

#include "core/ast_processors/okl_sema_processor/handlers/decl_ref_expr.h"
#include "core/ast_processors/okl_sema_processor/handlers/function.h"
#include "core/ast_processors/okl_sema_processor/handlers/loop.h"
#include "core/ast_processors/okl_sema_processor/handlers/recovery_expr.h"

#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

namespace {
using namespace clang;
using namespace oklt;

///////////////////// handlers entry points ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
HandleResult runPreValidationSemaStmt(const Attr* attr,
                                      const Stmt& stmt,
                                      OklSemaCtx& sema,
                                      SessionStage& stage) {
    switch (stmt.getStmtClass()) {
        case Stmt::DeclRefExprClass:
            return preValidateDeclRefExpr(cast<DeclRefExpr>(stmt), sema, stage);
        case Stmt::RecoveryExprClass:
            return preValidateRecoveryExpr(cast<RecoveryExpr>(stmt), sema, stage);
        default:
            return runDefaultPreActionStmt(attr, stmt, sema, stage);
    }
    return {};
}

HandleResult runPostValidationSemaStmt(const Attr* attr,
                                       const Stmt& stmt,
                                       OklSemaCtx& sema,
                                       SessionStage& stage) {
    switch (stmt.getStmtClass()) {
        case Stmt::DeclRefExprClass:
            return postValidateDeclRefExpr(cast<DeclRefExpr>(stmt), sema, stage);
        case Stmt::RecoveryExprClass:
            return postValidateRecoveryExpr(cast<RecoveryExpr>(stmt), sema, stage);
        default:
            return runDefaultPostActionStmt(attr, stmt, sema, stage);
    }
    return {};
}

__attribute__((constructor)) void registerAstNodeHanlder() {
    auto& mng = AstProcessorManager::instance();
    using DeclHandle = AstProcessorManager::DeclNodeHandle;
    using StmtHandle = AstProcessorManager::StmtNodeHandle;

    // default decl/stmt sema handlers
    auto ok = mng.registerDefaultHandle(
        {AstProcessorType::OKL_WITH_SEMA},
        DeclHandle{.preAction = runDefaultPreActionDecl, .postAction = runDefaultPostActionDecl});
    assert(ok);

    ok = mng.registerDefaultHandle(
        AstProcessorType::OKL_WITH_SEMA,
        StmtHandle{.preAction = runPreValidationSemaStmt, .postAction = runPostValidationSemaStmt});
    assert(ok);

    // sema for OKL kernel
    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, KERNEL_ATTR_NAME},
        DeclHandle{.preAction = makeSpecificSemaHandle(preValidateOklKernel),
                   .postAction = makeSpecificSemaHandle(postValidateOklKernel)});
    assert(ok);

    // all OKL attributed loop stmt
    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, TILE_ATTR_NAME},
        StmtHandle{.preAction = makeSpecificSemaHandle(preValidateOklForLoop),
                   .postAction = makeSpecificSemaHandle(postValidateOklForLoop)});
    assert(ok);

    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, OUTER_ATTR_NAME},
        StmtHandle{.preAction = makeSpecificSemaHandle(preValidateOklForLoop),
                   .postAction = makeSpecificSemaHandle(postValidateOklForLoop)});
    assert(ok);

    ok = mng.registerSpecificNodeHandle(
        {AstProcessorType::OKL_WITH_SEMA, INNER_ATTR_NAME},
        StmtHandle{.preAction = makeSpecificSemaHandle(preValidateOklForLoop),
                   .postAction = makeSpecificSemaHandle(postValidateOklForLoop)});
    assert(ok);
}
}  // namespace
