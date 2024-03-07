#include <oklt/core/kernel_metadata.h>

#include "core/attribute_manager/attribute_manager.h"
#include "core/attribute_manager/attributed_type_map.h"

#include "core/ast_processors/default_actions.h"
#include "core/ast_processors/okl_sema_processor/handlers/call_expr.h"
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
HandleResult preValidateCallExpr(const Attr* attr,
                                 const CallExpr& expr,
                                 OklSemaCtx& sema,
                                 SessionStage& stage) {
    return runDefaultPreActionStmt(attr, expr, sema, stage);
}

HandleResult postValidateCallExpr(const Attr* attr,
                                  const clang::CallExpr& expr,
                                  OklSemaCtx& sema,
                                  SessionStage& stage) {
    // If no errors, call default postValidate.
    if (!expr.containsErrors()) {
        return runDefaultPostActionStmt(attr, expr, sema, stage);
    }

    // Otherwise, we should try to call dim handler

    auto* args = expr.getArgs();
    if (expr.getNumArgs() == 0 || args == nullptr) {
        return {};
    }

    auto* dimVar = expr.getCallee();
    if (dimVar == nullptr) {
        return {};
    }

    auto declRefExpr = dyn_cast<DeclRefExpr>(dimVar);
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
