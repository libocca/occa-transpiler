#include "attributes/attribute_names.h"
#include "attributes/utils/default_handlers.h"
#include "core/handler_manager/backend_handler.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleSharedAttribute(SessionStage& s, const VarDecl& var, const Attr& a) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    auto varName = var.getNameAsString();
    // Desugar since it is attributed (since it is @shared variable)
    auto typeStr =
        QualType(var.getType().getTypePtr()->getUnqualifiedDesugaredType(), 0).getAsString();

    Error sharedError{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"};

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(sharedError);
    }
    auto* loopBelowInfo = loopInfo->getFirstAttributedChild();
    if (!loopBelowInfo || !(loopInfo->is(LoopType::Outer) && loopBelowInfo->is(LoopType::Inner))) {
        return tl::make_unexpected(sharedError);
    }

    auto newDeclaration =
        util::fmt(
            "auto & {} = "
            "*(sycl::ext::oneapi::group_local_memory_for_overwrite<{}>(item_.get_group()))",
            varName,
            typeStr)
            .value();

    SourceRange range(getAttrFullSourceRange(a).getBegin(), var.getSourceRange().getEnd());

    s.getRewriter().ReplaceText(range, newDeclaration);

    return defaultHandleSharedDeclAttribute(s, var, a);
}

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok = registerBackendHandler(TargetBackend::DPCPP, SHARED_ATTR_NAME, handleSharedAttribute);

    // Empty Stmt handler since @shared variable is of attributed type, it is called on DeclRefExpr
    ok &= registerBackendHandler(
        TargetBackend::DPCPP, SHARED_ATTR_NAME, defaultHandleSharedStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", SHARED_ATTR_NAME);
    }
}
}  // namespace
