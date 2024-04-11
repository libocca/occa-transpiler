#include "attributes/backend/serial/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPExclusiveHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, EXCLUSIVE_ATTR_NAME, ASTNodeKind::getFromNodeKind<DeclRefExpr>()},
        makeSpecificAttrHandle(serial_subset::handleExclusiveExprAttribute));
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, EXCLUSIVE_ATTR_NAME, ASTNodeKind::getFromNodeKind<VarDecl>()},
        makeSpecificAttrHandle(serial_subset::handleExclusiveDeclAttribute));

    if (!ok) {
        SPDLOG_ERROR("[SERIAL] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
