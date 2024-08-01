#include "attributes/backend/openmp/common.h"
#include "attributes/utils/default_handlers.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerOPENMPRestrictHandler() {
    auto ok = registerBackendHandler(
        TargetBackend::OPENMP, RESTRICT_ATTR_NAME, serial_subset::handleRestrictAttribute);

    ok &= registerBackendHandler(TargetBackend::OPENMP, RESTRICT_ATTR_NAME, emptyHandleStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENMP] Failed to register {} attribute handler", RESTRICT_ATTR_NAME);
    }
}
}  // namespace
