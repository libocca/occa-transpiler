#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/handler_manager/backend_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerBackendHandler(
        TargetBackend::HIP, ATOMIC_ATTR_NAME, cuda_subset::handleAtomicAttribute);

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
}  // namespace
