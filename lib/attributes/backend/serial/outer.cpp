#include "attributes/backend/serial/common.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPOuterHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, OUTER_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleOuterAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << OUTER_ATTR_NAME << " attribute handler (Serial)\n";
    }
}
}  // namespace
