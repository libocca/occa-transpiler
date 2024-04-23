#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/handler_manager/backend_handler.h"

#include <clang/AST/Attr.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;


__attribute__((constructor)) void registerAttrBackend() {
    auto ok = HandlerManager::registerBackendHandler(
        TargetBackend::DPCPP, EXCLUSIVE_ATTR_NAME, cuda_subset::handleExclusiveAttribute);

    ok &= HandlerManager::registerBackendHandler(
        TargetBackend::DPCPP, EXCLUSIVE_ATTR_NAME, defaultHandleExclusiveStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
