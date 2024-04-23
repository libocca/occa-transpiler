#include "attributes/utils/common.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/result.h"
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
HandleResult handleSharedAttribute(const clang::Attr& a, const clang::Decl& d, SessionStage& s) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    std::string replacedAttribute = " " + SHARED_MODIFIER + " ";

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!isLastOuter(loopInfo)) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }

    s.getRewriter().ReplaceText(getAttrFullSourceRange(a), replacedAttribute);
    return defaultHandleSharedDeclAttribute(a, d, s);
}
}  // namespace oklt::cuda_subset
