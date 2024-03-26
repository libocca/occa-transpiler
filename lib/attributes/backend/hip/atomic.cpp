#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, ATOMIC_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleAtomicAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << ATOMIC_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
