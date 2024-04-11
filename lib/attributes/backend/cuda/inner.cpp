#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerBackendHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, INNER_ATTR_NAME, ASTNodeKind::getFromNodeKind<ForStmt>()},
        makeSpecificAttrHandle(cuda_subset::handleInnerAttribute));

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register {} attribute handler", INNER_ATTR_NAME);
    }
}
}  // namespace
