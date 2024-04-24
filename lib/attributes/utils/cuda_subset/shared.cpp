#include "attributes/utils/default_handlers.h"
#include "core/handler_manager/result.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

#include <spdlog/spdlog.h>

namespace {
const std::string SHARED_MODIFIER = "__shared__";
}
namespace oklt::cuda_subset {
HandleResult handleSharedAttribute(SessionStage& s, const clang::Decl& d, const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    std::string replacedAttribute = " " + SHARED_MODIFIER + " ";

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (loopInfo && loopInfo->isRegular()) {
        loopInfo = loopInfo->getAttributedParent();
    }
    if (loopInfo && loopInfo->has(LoopType::Inner)) {
        return tl::make_unexpected(
            Error{{}, "Cannot define [@shared] variables inside an [@inner] loop"});
    }
    auto child = loopInfo ? loopInfo->getFirstAttributedChild() : nullptr;
    bool isInnerChild = child && child->has(LoopType::Inner);
    if (!loopInfo || !loopInfo->has(LoopType::Outer) || !isInnerChild) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }

    s.getRewriter().ReplaceText(getAttrFullSourceRange(a), replacedAttribute);

    return defaultHandleSharedDeclAttribute(s, d, a);
}
}  // namespace oklt::cuda_subset
