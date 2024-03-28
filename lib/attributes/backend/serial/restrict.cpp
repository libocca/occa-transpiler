#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPRestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, RESTRICT_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleRestrictAttribute));

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", RESTRICT_ATTR_NAME);
    }
}
}  // namespace
