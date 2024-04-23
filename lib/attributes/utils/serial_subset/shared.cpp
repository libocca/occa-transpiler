#include "attributes/utils/common.h"
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
    if (!isLastOuter(loopInfo)) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }
    removeAttribute(s, a);
    return defaultHandleSharedDeclAttribute(s, decl, a);
}

}  // namespace oklt::serial_subset
