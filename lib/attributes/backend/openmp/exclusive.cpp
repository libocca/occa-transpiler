#include "attributes/attribute_names.h"
#include "attributes/utils/serial_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

__attribute__((constructor)) void registerOPENMPExclusiveHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleExclusiveExprAttribute));
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(serial_subset::handleExclusiveDeclAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
