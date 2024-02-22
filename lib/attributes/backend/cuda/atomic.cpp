#include "attributes/attribute_names.h"
#include "attributes/utils/handle_atomic.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

bool handleCUDAAtomicAttribute(const clang::Attr* attr,
                               const clang::Stmt* stmt,
                               SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr->getNormalizedFullName() << '\n';
#endif
    return handleAtomicAttribute(attr, stmt, stage);
}

__attribute__((constructor)) void registerCUDAAtomicAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, ATOMIC_ATTR_NAME}, AttrStmtHandler{handleCUDAAtomicAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << ATOMIC_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
