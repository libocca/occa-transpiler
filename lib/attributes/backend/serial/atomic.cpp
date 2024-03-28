#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPAtomicHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, ATOMIC_ATTR_NAME},
        AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
}  // namespace
