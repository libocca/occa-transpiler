#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/handler_manager/backend_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerCUDAExclusiveAttrBackend() {
    auto ok = registerBackendHandler(
        TargetBackend::CUDA, EXCLUSIVE_ATTR_NAME, cuda_subset::handleExclusiveAttribute);
    ok &= registerBackendHandler(
        TargetBackend::CUDA, EXCLUSIVE_ATTR_NAME, defaultHandleSharedStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
