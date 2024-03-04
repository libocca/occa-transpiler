#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string restrictText = "__restrict__ ";

HandleResult handleOPENMPRestrictAttribute(const clang::Attr& a,
                                           const clang::VarDecl& decl,
                                           SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif

    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 2)
        .addReplacement(OKL_TRANSPILED_ARG, getAttrFullSourceRange(a), "")
        .addReplacement(OKL_TRANSPILED_ARG, decl.getLocation(), restrictText)
        .build();
}

__attribute__((constructor)) void registerOPENMPRestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, RESTRICT_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPRestrictAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace