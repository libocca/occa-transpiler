#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, TILE_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleTileAttribute));

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", TILE_ATTR_NAME);
    }
}
}  // namespace
