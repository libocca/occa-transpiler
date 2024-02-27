#include "attributes/frontend/params/tile.h"
#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;
HandleResult handleTileAttribute(const clang::Attr* a,
                                 const clang::ForStmt* forStmt,
                                 const TileParams* params,
                                 SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] DPCPP: Handle @tile.\n";
#endif
    return true;
}
__attribute__((constructor)) void registerDpcppTileAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, TILE_ATTR_NAME}, makeSpecificAttrHandle(handleTileAttribute));

    if (!ok) {
        llvm::errs() << "failed to register" << TILE_ATTR_NAME
                     << "attribute handler for DPCPP backend\n";
    }
}
}  // namespace
