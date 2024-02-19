#include "core/attribute_manager/attribute_manager.h"
#include "attributes/attribute_names.h"

#include "attributes/backend/common/cuda_subset/cuda_subset.h"

namespace {
using namespace oklt;
__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, TILE_ATTR_NAME}, AttrStmtHandler(cuda_subset::handleTileAttribute));

    if (!ok) {
        llvm::errs() << "failed to register tile attribute handler\n";
    }
}
}  // namespace
