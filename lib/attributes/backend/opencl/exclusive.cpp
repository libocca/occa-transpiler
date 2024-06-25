#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/handler_manager/backend_handler.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleExclusiveDeclAttribute(SessionStage& s , const Decl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute (decl)");

    removeAttribute(s, a);
    return {};
}

HandleResult handleExclusiveVarDeclAttribute(SessionStage& s , const VarDecl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute (decl)");

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@exclusive: failed to fetch loop meta data from sema"});
    }

    auto compStmt = dyn_cast_or_null<CompoundStmt>(loopInfo->stmt.getBody());
    if (!compStmt || !loopInfo->is(LoopType::Outer)) {
        return tl::make_unexpected(
            Error{{}, "Must define [@exclusive] variables between [@outer] and [@inner] loops"});
    }

    auto child = loopInfo->getFirstAttributedChild();
    if (!child || !child->is(LoopType::Inner)) {
        return tl::make_unexpected(
            Error{{}, "Must define [@exclusive] variables between [@outer] and [@inner] loops"});
    }

    removeAttribute(s, a);
    return {};
}

HandleResult handleExclusiveExprAttribute(SessionStage& s , const DeclRefExpr& expr,const Attr& a) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute (stmt)");

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@exclusive: failed to fetch loop meta data from sema"});
    }

    removeAttribute(s, a);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerBackendHandler(
        TargetBackend::OPENCL, EXCLUSIVE_ATTR_NAME, handleExclusiveExprAttribute);
    ok &= registerBackendHandler(
        TargetBackend::OPENCL, EXCLUSIVE_ATTR_NAME, handleExclusiveDeclAttribute);
    ok &= registerBackendHandler(
        TargetBackend::OPENCL, EXCLUSIVE_ATTR_NAME, handleExclusiveVarDeclAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
