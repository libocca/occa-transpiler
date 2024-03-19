#include "attributes/backend/serial/common.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPRestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, RESTRICT_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleRestrictAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME
                     << " attribute handler (Serial)\n";
    }
}
}  // namespace
