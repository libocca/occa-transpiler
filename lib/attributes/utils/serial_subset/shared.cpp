#include <oklt/core/kernel_metadata.h>

#include "attributes/utils/default_handlers.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt::serial_subset {
using namespace clang;

HandleResult handleSharedAttribute(SessionStage& s, const Decl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

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

    // This diagnostic is applied only to variable declaration
    // TODO: if var of type declared with @shared is not between @outer and @inner, error isnt risen
    if (!clang::isa<clang::TypeDecl>(decl)) {
        if (!loopInfo || !loopInfo->has(LoopType::Outer) || !isInnerChild) {
            return tl::make_unexpected(
                Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
        }
    } else {
        // Push warning that can't check that typedef @shared var is between outer and inner loop
        s.pushWarning("Using [@shared] with typedef doesn't have proper semantic validation yet");
    }

    removeAttribute(s, a);

    return defaultHandleSharedDeclAttribute(s, decl, a);
}

}  // namespace oklt::serial_subset
