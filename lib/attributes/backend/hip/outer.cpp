#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
__attribute__((constructor)) void registerHIPOuterAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, OUTER_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleOuterAttribute));

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register {} attribute handler", OUTER_ATTR_NAME);
    }
}
}  // namespace
