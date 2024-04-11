#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPBarrierHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, BARRIER_ATTR_NAME, ASTNodeKind::getFromNodeKind<Stmt>()},
        makeSpecificAttrHandle(serial_subset::handleEmptyStmtAttribute));

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", BARRIER_ATTR_NAME);
    }
}
}  // namespace
