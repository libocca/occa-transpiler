#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerCUDAExclusiveAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleExclusiveAttribute));
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(defaultHandleSharedStmtAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
