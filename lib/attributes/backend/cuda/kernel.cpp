#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/handler_manager/backend_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::HandlerManager::instance().registerBackendHandler(
        TargetBackend::CUDA, KERNEL_ATTR_NAME, cuda_subset::handleKernelAttribute);

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register {} attribute handler", KERNEL_ATTR_NAME);
    }
}
}  // namespace
