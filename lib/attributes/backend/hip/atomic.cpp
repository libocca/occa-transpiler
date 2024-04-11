#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, ATOMIC_ATTR_NAME, ASTNodeKind::getFromNodeKind<Stmt>()},
        makeSpecificAttrHandle(cuda_subset::handleAtomicAttribute));

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
}  // namespace
