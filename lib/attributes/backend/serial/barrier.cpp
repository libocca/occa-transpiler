#include "attributes/backend/serial/common.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPBarrierHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, BARRIER_ATTR_NAME},
        AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << BARRIER_ATTR_NAME
                     << " attribute handler (Serial)\n";
    }
}
}  // namespace
