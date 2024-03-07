#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

HandleResult handleCUDAExclusiveExprAttribute(const clang::Attr& a,
                                              const clang::DeclRefExpr& expr,
                                              SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    return {};
}

__attribute__((constructor)) void registerCUDAExclusiveAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleExclusiveAttribute));
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(handleCUDAExclusiveExprAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
