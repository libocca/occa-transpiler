#include "attributes/backend/serial/common.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPExclusiveHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleExclusiveExprAttribute));
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleExclusiveDeclAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME
                     << " attribute handler (Serial)\n";
    }
}
}  // namespace
