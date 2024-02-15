#include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>
#include <oklt/attributes/frontend/parsers/kernel.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>

namespace {
using namespace oklt;

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, KERNEL_ATTR_NAME},
        {parseKernelAttribute, cuda_subset::handleKernelAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME << " attribute handler (CUDA)\n";
    }
}
}  // namespace
