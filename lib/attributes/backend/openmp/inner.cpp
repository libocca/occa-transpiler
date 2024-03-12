#include "attributes/backend/openmp/common.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPOuterHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, INNER_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleInnerAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << INNER_ATTR_NAME << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
