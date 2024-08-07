#include "attributes/backend/serial/common.h"
#include "attributes/utils/default_handlers.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = registerBackendHandler(
        TargetBackend::SERIAL, SHARED_ATTR_NAME, serial_subset::handleSharedDeclAttribute);
    ok &= registerBackendHandler(
        TargetBackend::SERIAL, SHARED_ATTR_NAME, serial_subset::handleSharedTypeAttribute);
    ok &= registerBackendHandler(
        TargetBackend::SERIAL, SHARED_ATTR_NAME, serial_subset::handleSharedVarAttribute);

    // Empty Stmt handler since @shared variable is of attributed type, it is called on DeclRefExpr
    ok &= registerBackendHandler(
        TargetBackend::SERIAL, SHARED_ATTR_NAME, defaultHandleSharedStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", SHARED_ATTR_NAME);
    }
}
}  // namespace
