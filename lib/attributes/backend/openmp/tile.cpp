#include "attributes/backend/openmp/common.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string prefixText = "#pragma omp parallel for\n";

HandleResult handleOPENMPTileAttribute(const Attr& a,
                                       const ForStmt& stmt,
                                       const TileParams* params,
                                       SessionStage& s) {
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

    return serial_subset::handleTileAttribute(a, stmt, params, s);
}

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, TILE_ATTR_NAME}, makeSpecificAttrHandle(handleOPENMPTileAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << TILE_ATTR_NAME << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
