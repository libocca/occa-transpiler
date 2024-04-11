#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPAtomicHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, ATOMIC_ATTR_NAME, ASTNodeKind::getFromNodeKind<Stmt>()},
        makeSpecificAttrHandle(serial_subset::handleEmptyStmtAttribute));

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
}  // namespace
