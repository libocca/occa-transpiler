#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerCUDAExclusiveAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, EXCLUSIVE_ATTR_NAME, ASTNodeKind::getFromNodeKind<Decl>()},
        makeSpecificAttrHandle(cuda_subset::handleExclusiveAttribute));
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, EXCLUSIVE_ATTR_NAME, ASTNodeKind::getFromNodeKind<Stmt>()},
        makeSpecificAttrHandle(defaultHandleSharedStmtAttribute));

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
