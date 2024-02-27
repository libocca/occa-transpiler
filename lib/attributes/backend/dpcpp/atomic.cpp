#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

HandleResult handleAtomicAttribute(const clang::Attr* a,
                                   const clang::ForStmt* forStmt,
                                   SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] DPCPP: Handle @atomic.\n";
#endif
    return true;
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, ATOMIC_ATTR_NAME}, makeSpecificAttrHandle(handleAtomicAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << ATOMIC_ATTR_NAME
                     << " attribute handler for DPCPP backend\n";
    }
}
}  // namespace
