#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>

namespace {
using namespace oklt;
using namespace clang;
HandleResult handleBarrierAttribute(const clang::Attr& a,
                                    const clang::Stmt& stmt,
                                    SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] handle attribute: @barrier\n";
#endif
    auto& rewriter = s.getRewriter();
    SourceRange range(a.getRange().getBegin().getLocWithOffset(-2), stmt.getEndLoc());
    rewriter.ReplaceText(range, "item_.barrier(sycl::access::fence_space::local_space);");

    return true;
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, BARRIER_ATTR_NAME},
        makeSpecificAttrHandle(handleBarrierAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << BARRIER_ATTR_NAME
                     << " attribute handler (DPCPP)\n";
    }
}
}  // namespace
