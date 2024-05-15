#include "attributes/backend/metal/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerAttrBackend() {
    auto ok =
        registerBackendHandler(TargetBackend::METAL, ATOMIC_ATTR_NAME, emptyHandleStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
}  // namespace
