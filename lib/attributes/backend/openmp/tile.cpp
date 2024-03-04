#include "attributes/frontend/params/tile.h"
#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/ParentMapContext.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOPENMPTileAttribute(const Attr& attr,
                                       const ForStmt& stmt,
                                       const TileParams* params,
                                       SessionStage& stage) {
    removeAttribute(attr, stage);

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@tile: failed to fetch loop meta data from sema"});
    }

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @tile. Parsed for loop: Init("
                 << "type: " << loopInfo->type << ", name: " << forLoopMetaData->name
                 << ", initValue: " << loopInfo->range.start
                 << "), Cond(rhsExpr: " << loopInfo->range.end
                 << "), Inc(rhsInc: " << loopInfo->inc.val << ", isUnary: " << loopInfo->isUnary()
                 << ")\n";
#endif
    return {};
}

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, TILE_ATTR_NAME}, makeSpecificAttrHandle(handleOPENMPTileAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << TILE_ATTR_NAME << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
