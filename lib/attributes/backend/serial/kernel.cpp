#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPKernelHandler() {
    auto ok = registerBackendHandler(
        TargetBackend::SERIAL, KERNEL_ATTR_NAME, serial_subset::handleKernelAttribute);

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", KERNEL_ATTR_NAME);
    }
}
}  // namespace
