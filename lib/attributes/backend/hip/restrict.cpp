#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/handler_manager/backend_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok = registerBackendHandler(
        TargetBackend::HIP, RESTRICT_ATTR_NAME, cuda_subset::handleRestrictAttribute);
    ok &= registerBackendHandler(TargetBackend::HIP, RESTRICT_ATTR_NAME, emptyHandleStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register {} attribute handler", RESTRICT_ATTR_NAME);
    }
}
}  // namespace
