#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/empty_handlers.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, SHARED_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleSharedAttribute));

    ok = ok && oklt::AttributeManager::instance().registerBackendHandler(
                   {TargetBackend::HIP, SHARED_ATTR_NAME},
                   makeSpecificAttrHandle(emptyHandleSharedStmtAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << SHARED_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
