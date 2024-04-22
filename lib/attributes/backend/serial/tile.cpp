#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = HandlerManager::registerBackendHandler(
        TargetBackend::SERIAL, TILE_ATTR_NAME, serial_subset::handleTileAttribute);

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", TILE_ATTR_NAME);
    }
}
}  // namespace
