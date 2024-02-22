#include "core/attribute_manager/attribute_manager.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"

namespace {
using namespace oklt;
__attribute__((constructor)) void registerHIPTileAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, INNER_ATTR_NAME}, AttrStmtHandler{cuda_subset::handleInnerAttribute});

    if (!ok) {
        llvm::errs() << "failed to register tile attribute handler\n";
    }
}
}  // namespace
