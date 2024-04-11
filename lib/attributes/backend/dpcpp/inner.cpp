#include "attributes/attribute_names.h"
#include "attributes/backend/dpcpp/common.h"
#include "attributes/frontend/params/loop.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/kernel_utils.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleInnerAttribute(SessionStage& s,
                                  const clang::ForStmt& forStmt,
                                  const clang::Attr& a,
                                  const AttributedLoop* params) {
    SPDLOG_DEBUG("Handle [@inner] attribute");
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@inner params nullptr"});
    }

    auto& astCtx = s.getCompiler().getASTContext();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@inner: failed to fetch loop meta data from sema"});
    }

    auto updatedParams = *params;
    // Auto Axis in loopInfo are replaced with specific. TODO: maybe somehow update params earlier?
    updatedParams.axis = loopInfo->axis.front();

    int openedScopeCounter = 0;
    auto prefixCode = dpcpp::buildInnerOuterLoopIdxLine(
        *loopInfo, updatedParams, openedScopeCounter, s.getRewriter());
    auto suffixCode = buildCloseScopes(openedScopeCounter);
    std::string afterRBraceCode = "";
    if (loopInfo->shouldSync()) {
        afterRBraceCode += dpcpp::SYNC_THREADS_BARRIER + ";\n";
    }

    handleChildAttr(forStmt, NO_BARRIER_ATTR_NAME, s);

    return replaceAttributedLoop(s, forStmt, a, suffixCode, afterRBraceCode, prefixCode, true);
}

__attribute__((constructor)) void registerDpppInnerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, INNER_ATTR_NAME, ASTNodeKind::getFromNodeKind<ForStmt>()},
        makeSpecificAttrHandle(handleInnerAttribute));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", INNER_ATTR_NAME);
    }
}
}  // namespace
