#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;
HandleResult handleRestrictAttribute(const clang::Attr* a,
                                     const clang::Decl* d,
                                     SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] DPCPP: Handle @restrict.\n";
#endif
    return true;
}

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, RESTRICT_ATTR_NAME},
        makeSpecificAttrHandle(handleRestrictAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME
                     << " attribute handler for DPCPP backend\n";
    }
}
}  // namespace
