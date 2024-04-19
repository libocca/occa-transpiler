#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/handler_manager/backend_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerCUDABarrierAttrBackend() {
    auto ok = oklt::HandlerManager::instance().registerBackendHandler(
        TargetBackend::CUDA, BARRIER_ATTR_NAME, cuda_subset::handleBarrierAttribute);

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register {} attribute handler", BARRIER_ATTR_NAME);
    }
}
}  // namespace
