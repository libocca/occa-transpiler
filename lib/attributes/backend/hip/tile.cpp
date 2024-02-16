#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>
//#include "attributes/params/tile.hpp"
#include "attributes/utils/replace_attribute.h"

namespace {
using namespace oklt;
__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, TILE_ATTR_NAME},
        AttrStmtHandler {handleTileAttribute});

    if (!ok) {
        llvm::errs() << "failed to register tile attribute handler\n";
    }
}
}  // namespace
