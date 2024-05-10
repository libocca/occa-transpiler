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

std::string getCleanTypeString(SessionStage& s, QualType t) {
    static std::string annoTypeStr = " [[clang::annotate_type(...)]]";
    auto str = t.getAsString();

    auto pos = str.find(annoTypeStr);
    while (pos != std::string::npos) {
        str.replace(pos, annoTypeStr.size(), "");
        pos = str.find(annoTypeStr);
    }

    return str;
}

HandleResult handleSharedDeclAttribute(SessionStage& s, const Decl& var, const Attr& a) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    return removeAttribute(s, a);
}

HandleResult handleSharedTypeAttribute(SessionStage& s, const TypedefDecl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    removeAttribute(s, a);

    auto typeName = decl.getDeclName().getAsString();
    auto typeStr =
        getCleanTypeString(s, QualType(decl.getTypeForDecl()->getUnqualifiedDesugaredType(), 0));

    auto newDeclaration =
        util::fmt(
            "auto & {} = "
            "*(sycl::ext::oneapi::group_local_memory_for_overwrite<typedef {}>(item_.get_group()))",
            typeName,
            typeStr)
            .value();

    s.getRewriter().ReplaceText(decl.getSourceRange(), newDeclaration);

    return {};
}

HandleResult handleSharedVarAttribute(SessionStage& s, const VarDecl& var, const Attr& a) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    removeAttribute(s, a);

    auto varName = var.getNameAsString();
    auto typeStr = getCleanTypeString(
        s, QualType(var.getType().getTypePtr()->getUnqualifiedDesugaredType(), 0));

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

    auto newDeclaration =
        util::fmt(
            "auto & {} = "
            "*(sycl::ext::oneapi::group_local_memory_for_overwrite<{}>(item_.get_group()))",
            varName,
            typeStr)
            .value();

    SourceRange range(var.getBeginLoc(), var.getSourceRange().getEnd());

    s.getRewriter().ReplaceText(range, newDeclaration);

    return defaultHandleSharedDeclAttribute(s, var, a);
}

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok =
        registerBackendHandler(TargetBackend::DPCPP, SHARED_ATTR_NAME, handleSharedDeclAttribute);
    ok &= registerBackendHandler(TargetBackend::DPCPP, SHARED_ATTR_NAME, handleSharedTypeAttribute);
    ok &= registerBackendHandler(TargetBackend::DPCPP, SHARED_ATTR_NAME, handleSharedVarAttribute);

    // Empty Stmt handler since @shared variable is of attributed type, it is called on DeclRefExpr
    ok &= registerBackendHandler(
        TargetBackend::DPCPP, SHARED_ATTR_NAME, defaultHandleSharedStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", SHARED_ATTR_NAME);
    }
}
}  // namespace
