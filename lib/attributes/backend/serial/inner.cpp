#include "attributes/backend/serial/common.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPOuterHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, INNER_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleInnerAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << INNER_ATTR_NAME << " attribute handler (Serial)\n";
    }
}
}  // namespace
