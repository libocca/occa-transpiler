#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

HandleResult handleHIPExclusiveExprAttribute(const clang::Attr& a,
                                             const clang::DeclRefExpr& expr,
                                             SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    return {};
}

__attribute__((constructor)) void registerHIPExclusiveAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleExclusiveAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME << " attribute handler\n";
    }

    ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(handleHIPExclusiveExprAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
