#include <oklt/core/kernel_metadata.h>

#include "core/ast_processors/default_actions.h"
#include "core/ast_processors/okl_sema_processor/handlers/loop.h"
#include "core/attribute_manager/attribute_manager.h"
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
HandleResult preValidateOklForLoop(const Attr& attr,
                                   const ForStmt& stmt,
                                   OklSemaCtx& sema,
                                   SessionStage& stage) {
    auto params = stage.getAttrManager().parseAttr(attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    auto result = sema.validateOklForLoopOnPreTraverse(attr, stmt, &params.value());
    if (!result) {
        return tl::make_unexpected(std::move(result.error()));
    }

    return {};
}

HandleResult postValidateOklForLoop(const Attr& attr,
                                    const clang::ForStmt& stmt,
                                    OklSemaCtx& sema,
                                    SessionStage& stage) {
    auto result = runDefaultPostActionStmt(&attr, stmt, sema, stage);
    if (!result) {
        return result;
    }

    auto params = stage.getAttrManager().parseAttr(attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    auto ok = sema.validateOklForLoopOnPostTraverse(attr, stmt, &params.value());
    if (!ok) {
        //  make approptiate error code
        return tl::make_unexpected(std::move(ok.error()));
    }

    return result;
}

}  // namespace oklt
