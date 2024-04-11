#include "attributes/backend/openmp/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPRestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, RESTRICT_ATTR_NAME, ASTNodeKind::getFromNodeKind<Decl>()},
        makeSpecificAttrHandle(serial_subset::handleRestrictAttribute));

    if (!ok) {
        SPDLOG_ERROR("[OPENMP] Failed to register {} attribute handler", RESTRICT_ATTR_NAME);
    }
}
}  // namespace
