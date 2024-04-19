#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, RESTRICT_ATTR_NAME},
        makeSpecificAttrHandle(cuda_subset::handleRestrictAttribute));

    // allow @restrict in return type case
    ok &= oklt::AttributeManager::instance().registerCompatibleImplicitAttributePair(
        {TargetBackend::CUDA, clang::Decl::Kind::Function},
        {TargetBackend::CUDA, RESTRICT_ATTR_NAME});

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register {} attribute handler", RESTRICT_ATTR_NAME);
    }
}
}  // namespace
