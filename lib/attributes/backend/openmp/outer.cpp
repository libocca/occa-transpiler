#include "attributes/backend/openmp/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string prefixText = "\n#pragma omp parallel for\n";

HandleResult handleOPENMPOuterAttribute(SessionStage& s,
                                        const ForStmt& stmt,
                                        const Attr& a,
                                        const AttributedLoop* params) {
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@tile: failed to fetch loop meta data from sema"});
    }

    // Top level `@outer` loop
    auto parent = loopInfo->getAttributedParent();
    if (!parent && loopInfo->has(LoopType::Outer)) {
        s.getRewriter().InsertText(stmt.getBeginLoc(), prefixText, false, true);
    }

    return serial_subset::handleOuterAttribute(s, stmt, a, params);
}

__attribute__((constructor)) void registerOPENMPOuterHandler() {
    auto ok = HandlerManager::registerBackendHandler(
        TargetBackend::OPENMP, OUTER_ATTR_NAME, handleOPENMPOuterAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENMP] Failed to register {} attribute handler", OUTER_ATTR_NAME);
    }
}
}  // namespace
