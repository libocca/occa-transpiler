#include <clang/AST/Decl.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace {
using namespace oklt;

bool handleGlobalConstant(const clang::Decl* d, SessionStage& s) {
    return true;
}

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::CUDA, clang::Decl::Kind::Var}, DeclHandler{handleGlobalConstant});

    if (!ok) {
        llvm::errs() << "failed to register implicit handler for global constant\n";
    }
}
}  // namespace
