#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;
__attribute__((constructor)) void registerHIPTileAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, TILE_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleTileAttribute));

    if (!ok) {
        llvm::errs() << "failed to register tile attribute handler\n";
    }
}
}  // namespace
