#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/attribute_manager.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleExclusiveAttribute));
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(defaultHandleExclusiveStmtAttribute));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
