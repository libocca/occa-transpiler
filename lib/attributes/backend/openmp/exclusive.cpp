#include "attributes/attribute_names.h"
#include "attributes/utils/serial_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPExclusiveHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME, ASTNodeKind::getFromNodeKind<DeclRefExpr>()},
        makeSpecificAttrHandle(serial_subset::handleExclusiveExprAttribute));
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME, ASTNodeKind::getFromNodeKind<VarDecl>()},
        makeSpecificAttrHandle(serial_subset::handleExclusiveDeclAttribute));

    if (!ok) {
        SPDLOG_ERROR("[OPENMP] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
