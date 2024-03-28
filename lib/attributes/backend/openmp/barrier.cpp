#include "attributes/backend/openmp/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOPENMPBarrierAttribute(const Attr& a, const NullStmt& stmt, SessionStage& s) {
    SPDLOG_DEBUG("Handle [@barrier] attribute");

    SourceRange range(getAttrFullSourceRange(a).getBegin(), stmt.getEndLoc());
    s.getRewriter().RemoveText(range);
    return {};
}

__attribute__((constructor)) void registerOPENMPBarrierHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, BARRIER_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPBarrierAttribute));

    if (!ok) {
        SPDLOG_ERROR("[OPENMP] Failed to register {} attribute handler", BARRIER_ATTR_NAME);
    }
}
}  // namespace
