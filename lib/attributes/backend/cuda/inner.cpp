#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerBackendHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, INNER_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleInnerAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << INNER_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
