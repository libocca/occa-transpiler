#include "attributes/backend/serial/common.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, KERNEL_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleKernelAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME
                     << " attribute handler (Serial)\n";
    }
}
}  // namespace
