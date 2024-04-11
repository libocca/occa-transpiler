#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, SHARED_ATTR_NAME, ASTNodeKind::getFromNodeKind<Decl>()},
        makeSpecificAttrHandle(cuda_subset::handleSharedAttribute));

    // Empty Stmt handler since @shared variable is of attributed type, it is called on DeclRefExpr
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, SHARED_ATTR_NAME, ASTNodeKind::getFromNodeKind<Stmt>()},
        makeSpecificAttrHandle(defaultHandleSharedStmtAttribute));

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register {} attribute handler", SHARED_ATTR_NAME);
    }
}
}  // namespace
