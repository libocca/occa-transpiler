#include "core/attribute_manager/attribute_manager.h"
#include "core/attribute_manager/attributed_type_map.h"

#include "core/ast_processors/default_actions.h"
#include "core/ast_processors/okl_sema_processor/handlers/decl_ref_expr.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>

namespace oklt {
using namespace clang;

HandleResult preValidateDeclRefExpr(const DeclRefExpr& expr,
                                    OklSemaCtx& sema,
                                    SessionStage& stage) {
    return {};
}

HandleResult postValidateDeclRefExpr(const DeclRefExpr& expr,
                                     OklSemaCtx& sema,
                                     SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto& attrTypeMap = stage.tryEmplaceUserCtx<AttributedTypeMap>();
    auto attrs = attrTypeMap.get(ctx, expr.getType());

    auto& am = stage.getAttrManager();
    auto expectedAttr = am.checkAttrs(attrs, expr, stage);
    if (!expectedAttr) {
        // TODO report diagnostic error using clang tooling
        return tl::make_unexpected(std::move(expectedAttr.error()));
    }

    return runDefaultPostActionStmt(expectedAttr.value(), expr, sema, stage);
}

}  // namespace oklt
