#include "attributes/attribute_names.h"
#include "attributes/utils/serial_subset/handle.h"
#include "core/handler_manager/backend_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPExclusiveHandler() {
    auto ok = registerBackendHandler(
        TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME, serial_subset::handleExclusiveExprAttribute);
    ok &= registerBackendHandler(
        TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME, serial_subset::handleExclusiveDeclAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENMP] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
