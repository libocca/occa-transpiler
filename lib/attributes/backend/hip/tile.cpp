#include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>
#include <oklt/attributes/frontend/parsers/tile.h>

namespace {
using namespace oklt;
__attribute__((constructor)) void registerTileHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, TILE_ATTR_NAME},
        {parseTileAttribute, cuda_subset::handleTileAttribute});

    if (!ok) {
        llvm::errs() << "failed to register tile attribute handler\n";
    }
}
}  // namespace
