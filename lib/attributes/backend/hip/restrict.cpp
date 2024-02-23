#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, RESTRICT_ATTR_NAME},
        AttrDeclHandler{cuda_subset::handleRestrictAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
