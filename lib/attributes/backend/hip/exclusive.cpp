#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/handler_manager/backend_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerHIPExclusiveAttrBackend() {
    auto ok = oklt::HandlerManager::instance().registerBackendHandler(
        TargetBackend::HIP, EXCLUSIVE_ATTR_NAME, cuda_subset::handleExclusiveAttribute);

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }

    ok = oklt::HandlerManager::instance().registerBackendHandler(
        TargetBackend::HIP, EXCLUSIVE_ATTR_NAME, defaultHandleExclusiveStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
