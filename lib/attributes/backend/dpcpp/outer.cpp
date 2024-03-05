#include <attributes/utils/code_gen.h>
#include "attributes/attribute_names.h"
#include "attributes/backend/dpcpp/common.h"
#include "attributes/frontend/params/loop.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOuterAttribute(const clang::Attr& a,
                                  const clang::ForStmt& forStmt,
                                  const AttributedLoop* params,
                                  SessionStage& s) {
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@outer params nullptr"});
    }

    auto& astCtx = s.getCompiler().getASTContext();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@outer: failed to fetch loop meta data from sema"});
    }

    int openedScopeCounter = 0;
    auto prefixCode = dpcpp::buildInnerOuterLoopIdxLine(*loopInfo, *params, openedScopeCounter);
    auto suffixCode = buildCloseScopes(openedScopeCounter);

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @outer attribute\n";
#endif
    return replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);
}

__attribute__((constructor)) void registerDpcppOuterAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, OUTER_ATTR_NAME}, makeSpecificAttrHandle(handleOuterAttribute));

    if (!ok) {
        llvm::errs() << "failed to register inner attribute handler\n";
    }
}
}  // namespace
