#include "attributes/backend/metal/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleAtomicStmtAttribute(SessionStage& s, const Stmt& stmt, const Attr& a) {
    SPDLOG_DEBUG("Handle attribute [{}]", a.getNormalizedFullName());

    removeAttribute(s, a);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok =
        registerBackendHandler(TargetBackend::METAL, ATOMIC_ATTR_NAME, handleAtomicStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
}  // namespace
