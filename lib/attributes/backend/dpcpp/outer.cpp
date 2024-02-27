#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOuterAttribute(const clang::Attr* a,
                                  const clang::ForStmt* forStmt,
                                  //   const AttributedLoop* params,
                                  SessionStage& s) {
    // auto& astCtx = s.getCompiler().getASTContext();
    // auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    // auto forLoopMetaData = sema.getLoopMetaData(forStmt);
    // if (!forLoopMetaData) {
    //     s.pushError(std::error_code(), "@tile: failed to fetch loop meta data from sema");
    //     return false;
    // }

    // int openedScopeCounter = 0;
    // auto prefixCode = inner_outer::buildInnerOuterLoopIdxLine(
    //     forLoopMetaData.value(), *params, openedScopeCounter);
    // auto suffixCode = buildCloseScopes(openedScopeCounter);

    // replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);
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
