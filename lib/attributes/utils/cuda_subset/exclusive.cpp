#include "attributes/utils/default_handlers.h"
#include "core/handler_manager/result.h"
#include "core/sema/okl_sema_ctx.h"
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
    if (loopInfo && loopInfo->isRegular()) {
        loopInfo = loopInfo->getAttributedParent();
    }
    if (loopInfo && loopInfo->has(LoopType::Inner)) {
        return tl::make_unexpected(
            Error{{}, "Cannot define [@exclusive] variables inside an [@inner] loop"});
    }
    auto child = loopInfo ? loopInfo->getFirstAttributedChild() : nullptr;
    bool isInnerChild = child && child->has(LoopType::Inner);
    if (!clang::isa<clang::TypeDecl>(decl)) {
        if (!loopInfo || !loopInfo->has(LoopType::Outer) || !isInnerChild) {
            return tl::make_unexpected(Error{
                {}, "Must define [@exclusive] variables between [@outer] and [@inner] loops"});
        }
    } else {
        // Push warning that can't check that typedef @exclusive var is between outer and inner loop
        stage.pushWarning(
            "Using [@exclusive] with typedef doesn't have proper semantic validation yet");
    }

    stage.getRewriter().RemoveText(getAttrFullSourceRange(attr));
    return defaultHandleExclusiveDeclAttribute(stage, decl, attr);
}
}  // namespace oklt::cuda_subset
