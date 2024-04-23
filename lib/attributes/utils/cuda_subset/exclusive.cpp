#include "attributes/utils/common.h"
#include "attributes/utils/default_handlers.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/handler_manager/result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

#include <spdlog/spdlog.h>

namespace oklt::cuda_subset {
HandleResult handleExclusiveAttribute(SessionStage& stage,
                                      const clang::Decl& decl,
                                      const clang::Attr& attr) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute");

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@exclusive: failed to fetch loop meta data from sema"});
    }

    auto definedBetweenOuterInner = isLastOuter(loopInfo);
    if (!definedBetweenOuterInner) {
        return tl::make_unexpected(
            Error{{}, "Must define [@exclusive] variables between [@outer] and [@inner] loops"});
    }

    stage.getRewriter().RemoveText(getAttrFullSourceRange(attr));
    return defaultHandleExclusiveDeclAttribute(stage, decl, attr);
}
}  // namespace oklt::cuda_subset
