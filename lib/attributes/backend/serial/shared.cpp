#include "attributes/backend/serial/common.h"
#include "attributes/utils/default_handlers.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, SHARED_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleSharedAttribute));

    // Empty Stmt hanler since @shared variable is of attributed type, it is called on DeclRefExpr
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, SHARED_ATTR_NAME},
        makeSpecificAttrHandle(defaultHandleSharedStmtAttribute));

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", SHARED_ATTR_NAME);
    }
}
}  // namespace
