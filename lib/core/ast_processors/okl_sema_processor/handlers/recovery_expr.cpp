#include <oklt/core/kernel_metadata.h>

#include "core/attribute_manager/attribute_manager.h"
#include "core/attribute_manager/attributed_type_map.h"

#include "core/ast_processors/default_actions.h"
#include "core/ast_processors/okl_sema_processor/handlers/recovery_expr.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>
#include <clang/AST/Attr.h>

#define OKL_SEMA_DEBUG
namespace {
using namespace oklt;
using namespace clang;
}  // namespace

namespace oklt {
HandleResult preValidateRecoveryExpr(const RecoveryExpr& expr,
                                     OklSemaCtx& sema,
                                     SessionStage& stage) {
    return {};
}

HandleResult postValidateRecoveryExpr(const clang::RecoveryExpr& expr,
                                      OklSemaCtx& sema,
                                      SessionStage& stage) {
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
        return tl::make_unexpected(std::move(expectedAttr.error()));
    }

    return runDefaultPostActionStmt(expectedAttr.value(), expr, sema, stage);
}

}  // namespace oklt
