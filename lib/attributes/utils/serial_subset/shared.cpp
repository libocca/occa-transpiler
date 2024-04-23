#include "attributes/utils/common.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>
namespace oklt::serial_subset {
using namespace clang;

HandleResult handleSharedAttribute(const Attr& a, const Decl& decl, SessionStage& s) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!isLastOuter(loopInfo)) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }
    removeAttribute(a, s);
    return defaultHandleSharedDeclAttribute(a, decl, s);
}

}  // namespace oklt::serial_subset
