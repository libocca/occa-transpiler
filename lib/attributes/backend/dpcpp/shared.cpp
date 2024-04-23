#include "attributes/attribute_names.h"
#include "attributes/utils/common.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleSharedAttribute(const Attr& a, const VarDecl& var, SessionStage& s) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    auto varName = var.getNameAsString();
    // Desugar since it is attributed (since it is @shared variable)
    auto typeStr =
        QualType(var.getType().getTypePtr()->getUnqualifiedDesugaredType(), 0).getAsString();

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!isLastOuter(loopInfo)) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
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

    return defaultHandleSharedDeclAttribute(a, var, s);
}

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, SHARED_ATTR_NAME}, makeSpecificAttrHandle(handleSharedAttribute));

    // Empty Stmt hanler since @shared variable is of attributed type, it is called on DeclRefExpr
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, SHARED_ATTR_NAME},
        makeSpecificAttrHandle(defaultHandleSharedStmtAttribute));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", SHARED_ATTR_NAME);
    }
}
}  // namespace
