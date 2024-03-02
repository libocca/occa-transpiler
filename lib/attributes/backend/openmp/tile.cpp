#include "attributes/frontend/params/tile.h"
#include "attributes/attribute_names.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/ParentMapContext.h>

namespace {
using namespace oklt;
using namespace clang;

bool isRootLevel(ParentMapContext& ctx, OklSemaCtx& sema, const DynTypedNode node) {
    auto parents = ctx.getParents(node);
    while (!parents.empty()) {
        if (auto funcStmt = parents[0].get<FunctionDecl>()) {
            return true;
        }

        if (auto forStmt = parents[0].get<ForStmt>()) {
            auto metadata = sema.getLoopMetaData(*forStmt);
            if (metadata.has_value()) {
                return false;
            }
        };

        parents = ctx.getParents(parents[0]);
    }

    return true;
}

HandleResult handleOPENMPTileAttribute(const Attr& attr,
                                       const ForStmt& stmt,
                                       const TileParams* params,
                                       SessionStage& stage) {
    removeAttribute(attr, stage);

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto forLoopMetaData = sema.getLoopMetaData(stmt);
    if (!forLoopMetaData) {
        return tl::make_unexpected(Error{{}, "@tile: failed to fetch loop meta data from sema"});
    }

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @tile. Parsed for loop: Init("
                 << "type: " << forLoopMetaData->type << ", name: " << forLoopMetaData->name
                 << ", initValue: " << forLoopMetaData->range.start
                 << "), Cond(rhsExpr: " << forLoopMetaData->range.end
                 << "), Inc(rhsInc: " << forLoopMetaData->inc.val
                 << ", isUnary: " << forLoopMetaData->isUnary() << ")\n";
#endif
    return true;
}

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, TILE_ATTR_NAME}, makeSpecificAttrHandle(handleOPENMPTileAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << TILE_ATTR_NAME << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
