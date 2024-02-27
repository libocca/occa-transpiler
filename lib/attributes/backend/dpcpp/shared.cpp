#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;
HandleResult handleSharedAttribute(const clang::Attr* a,
                                   const clang::ForStmt* forStmt,
                                   SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] DPCPP: Handle @shared.\n";
#endif
    return true;
}

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, SHARED_ATTR_NAME}, makeSpecificAttrHandle(handleSharedAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << SHARED_ATTR_NAME
                     << " attribute handler for DPCPP backend\n";
    }
}
}  // namespace
