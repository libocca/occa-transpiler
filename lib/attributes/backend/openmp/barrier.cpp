#include "attributes/backend/openmp/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOPENMPBarrierAttribute(SessionStage& s, const NullStmt& stmt, const Attr& a) {
    SPDLOG_DEBUG("Handle [@barrier] attribute");

    SourceRange range(getAttrFullSourceRange(a).getBegin(), stmt.getEndLoc());
    s.getRewriter().RemoveText(range);
    return {};
}

__attribute__((constructor)) void registerOPENMPBarrierHandler() {
    auto ok = registerBackendHandler(
        TargetBackend::OPENMP, BARRIER_ATTR_NAME, handleOPENMPBarrierAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENMP] Failed to register {} attribute handler", BARRIER_ATTR_NAME);
    }
}
}  // namespace
