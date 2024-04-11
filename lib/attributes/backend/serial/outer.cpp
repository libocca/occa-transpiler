#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPOuterHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, OUTER_ATTR_NAME, ASTNodeKind::getFromNodeKind<ForStmt>()},
        makeSpecificAttrHandle(serial_subset::handleOuterAttribute));

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", OUTER_ATTR_NAME);
    }
}
}  // namespace
