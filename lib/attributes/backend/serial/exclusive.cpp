#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPExclusiveHandler() {
    auto ok = registerBackendHandler(
        TargetBackend::SERIAL, EXCLUSIVE_ATTR_NAME, serial_subset::handleExclusiveExprAttribute);
    ok &= registerBackendHandler(
        TargetBackend::SERIAL, EXCLUSIVE_ATTR_NAME, serial_subset::handleExclusiveDeclAttribute);
    ok &= registerBackendHandler(
        TargetBackend::SERIAL, EXCLUSIVE_ATTR_NAME, serial_subset::handleExclusiveVarAttribute);

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
