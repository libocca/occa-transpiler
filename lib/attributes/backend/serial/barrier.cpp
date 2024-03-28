#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPBarrierHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, BARRIER_ATTR_NAME},
        AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", BARRIER_ATTR_NAME);
    }
}
}  // namespace
