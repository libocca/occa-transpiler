#include <oklt/core/attribute_manager/attribute_manager.h>
#include "attributes/utils/replace_attribute.h"

namespace {
using namespace oklt;

bool handleGlobalFunction(const clang::Decl* decl, oklt::SessionStage& s) {
    const std::string HIP_FUNCTION_QUALIFIER = "__device__";
    return oklt::handleGlobalFunction(decl, s, HIP_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, clang::Decl::Kind::Function}, DeclHandler{handleGlobalFunction});

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for global function (HIP)\n";
    }
}
}  // namespace
