#include "attributes/backend/serial/common.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, SHARED_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleSharedAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << SHARED_ATTR_NAME
                     << " attribute handler (Serial)\n";
    }
}
}  // namespace
