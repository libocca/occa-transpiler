#include <attributes/utils/code_gen.h>
#include "attributes/attribute_names.h"
#include "attributes/backend/dpcpp/common.hpp"
#include "attributes/frontend/params/loop.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOuterAttribute(const clang::Attr* a,
                                  const clang::ForStmt* forStmt,
                                  const AttributedLoop* params,
                                  SessionStage& s) {
    auto& astCtx = s.getCompiler().getASTContext();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto forLoopMetaData = sema.getLoopMetaData(forStmt);
    if (!forLoopMetaData) {
        return tl::make_unexpected(Error{{}, "@outer: failed to fetch loop meta data from sema"});
    }

    int openedScopeCounter = 0;
    auto prefixCode =
        dpcpp::buildInnerOuterLoopIdxLine(forLoopMetaData.value(), *params, openedScopeCounter);
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @outer attribute\n";
#endif
    return true;
}

__attribute__((constructor)) void registerDpcppOuterAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, OUTER_ATTR_NAME}, makeSpecificAttrHandle(handleOuterAttribute));

    if (!ok) {
        llvm::errs() << "failed to register inner attribute handler\n";
    }
}
}  // namespace
