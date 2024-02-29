#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"

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

    SourceRange range(a.getRange().getBegin().getLocWithOffset(-2), stmt.getEndLoc());
    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1)
        .addReplacement(
            OKL_BARRIER, range, "item_.barrier(sycl::access::fence_space::local_space);")
        .build();
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, BARRIER_ATTR_NAME}, makeSpecificAttrHandle(handleBarrierAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << BARRIER_ATTR_NAME
                     << " attribute handler (DPCPP)\n";
    }
}
}  // namespace
