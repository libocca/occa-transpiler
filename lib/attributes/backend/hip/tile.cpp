#include "core/attribute_manager/attribute_manager.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/replace_attribute.h"

namespace {
using namespace oklt;
__attribute__((constructor)) void registerHIPTileAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, TILE_ATTR_NAME}, AttrStmtHandler{handleTileAttribute});

    if (!ok) {
        llvm::errs() << "failed to register tile attribute handler\n";
    }
}
}  // namespace
