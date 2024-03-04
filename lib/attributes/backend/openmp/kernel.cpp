#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string externC = "extern \"C\"";

HandleResult handleOPENMPKernelAttribute(const clang::Attr& attr,
                                         const clang::FunctionDecl& decl,
                                         SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif

    auto trans = TranspilationBuilder(stage.getCompiler().getSourceManager(),
                                      attr.getNormalizedFullName(),
                                      1 + decl.param_size());

    // Add 'extern "C"'
    SourceRange attr_range = getAttrFullSourceRange(attr);
    trans.addReplacement(OKL_TRANSPILED_ATTR, attr_range, externC);

    // Convert ann non-pointer params to references
    auto& ctx = stage.getCompiler().getASTContext();
    for (const auto param : decl.parameters()) {
        if (!param || !param->getType().getTypePtrOrNull()) {
            continue;
        }

        auto t = param->getType();
        if (!t->isPointerType()) {
            auto locRange = param->DeclaratorDecl::getSourceRange();
            trans.addReplacement(OKL_TRANSPILED_ARG, locRange.getEnd(), " &");
        }
    }

    return trans.build();
}

__attribute__((constructor)) void registerOPENMPKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, KERNEL_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPKernelAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << KERNEL_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
