#include "core/attribute_manager/attribute_manager.h"
#include "attributes/attribute_names.h"

namespace {
using namespace oklt;

bool handleKernelAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
    return true;
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, KERNEL_ATTR_NAME}, AttrDeclHandler{handleKernelAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME << " attribute handler (CUDA)\n";
    }
}
}  // namespace
