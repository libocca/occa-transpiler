#include "attributes/attribute_names.h"
#include "attributes/backend/dpcpp/common.h"
#include "attributes/frontend/params/loop.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/kernel_utils.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleInnerAttribute(const clang::Attr& a,
                                  const clang::ForStmt& forStmt,
                                  const AttributedLoop* params,
                                  SessionStage& s) {
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

    handleChildAttr(forStmt, NOBARRIER_ATTR_NAME, s);
  
    if (loopInfo->shouldSync()) {
        suffixCode += dpcpp::SYNC_THREADS_BARRIER + ";\n";
    }
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @inner attribute\n";
#endif

    return replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s, true);
}

__attribute__((constructor)) void registerDpppInnerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, INNER_ATTR_NAME}, makeSpecificAttrHandle(handleInnerAttribute));

    if (!ok) {
        llvm::errs() << "failed to register tile attribute handler\n";
    }
}
}  // namespace
