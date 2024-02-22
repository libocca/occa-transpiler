#include <oklt/core/kernel_metadata.h>

#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>
#include <clang/AST/Attr.h>

#define OKL_SEMA_DEBUG
namespace {
using namespace oklt;
using namespace clang;
}  // namespace

namespace oklt {
bool preValidateOklForLoopSema(const Attr* attr,
                        const ForStmt* stmt,
                        SessionStage& stage,
                        OklSemaCtx& sema) {
    auto result = sema.validateOklForLoopOnPreTraverse(attr, stmt);
    if (!result) {
        //  make approptiate error code
        stage.pushError(result.error());
        return false;
    }

    return true;
}

bool postValidateOklForLoopSema(const Attr* attr,
                         const clang::ForStmt* stmt,
                         SessionStage& stage,
                         OklSemaCtx& sema) {
    auto result = sema.validateOklForLoopOnPostTraverse(attr, stmt);
    if (!result) {
        //  make approptiate error code
        stage.pushError(result.error());
        return false;
    }

    return true;
}

}  // namespace oklt
