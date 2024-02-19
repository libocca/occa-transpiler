#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>

namespace {
using namespace oklt;

bool handleHIPKernelAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
    return true;
}

__attribute__((constructor)) void registerHIPKernelAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::HIP, KERNEL_ATTR_NAME}, AttrDeclHandler{handleHIPKernelAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME << " attribute handler (CUDA)\n";
    }
}
}  // namespace
