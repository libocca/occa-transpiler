#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>
#include <oklt/attributes/frontend/parsers/tile.hpp>
#include "attributes/utils/replace_attribute.h"

namespace {
using namespace oklt;
__attribute__((constructor)) void registerTileHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, TILE_ATTR_NAME},
        {parseTileAttribute, handleTileAttribute});

    if (!ok) {
        llvm::errs() << "failed to register tile attribute handler\n";
    }
}
}  // namespace
