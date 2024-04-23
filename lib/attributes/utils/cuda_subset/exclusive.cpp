#include "attributes/utils/common.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/result.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

#include <spdlog/spdlog.h>

namespace oklt::cuda_subset {
HandleResult handleExclusiveAttribute(const clang::Attr& attr,
                                      const clang::Decl& decl,
                                      SessionStage& stage) {
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
    return defaultHandleExclusiveDeclAttribute(attr, decl, stage);
}
}  // namespace oklt::cuda_subset
