#include "attributes/backend/serial/common.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPAtomicHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::SERIAL, ATOMIC_ATTR_NAME},
        AttrStmtHandler{serial_subset::handleEmptyStmtAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << ATOMIC_ATTR_NAME
                     << " attribute handler (Serial)\n";
    }
}
}  // namespace
