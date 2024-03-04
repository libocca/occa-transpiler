#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string prefixExprText = "#pragma omp atomic\n";
const std::string prefixCompoundText = "#pragma omp critical\n";

HandleResult handleOPENMPAtomicAttribute(const Attr& a, const Stmt& stmt, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    SourceRange attr_range = getAttrFullSourceRange(a);
    if (isa<Expr>(stmt)) {
        return TranspilationBuilder(
                   s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1)
            .addReplacement(OKL_TRANSPILED_ATTR, attr_range, prefixExprText)
            .build();
    }
    if (isa<CompoundStmt>(stmt)) {
        return TranspilationBuilder(
                   s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1)
            .addReplacement(OKL_TRANSPILED_ATTR, attr_range, prefixCompoundText)
            .build();
    }

    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1)
        .addReplacement(OKL_TRANSPILED_ATTR, attr_range, "")
        .build();
}

__attribute__((constructor)) void registerOPENMPAtomicHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, ATOMIC_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPAtomicAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << ATOMIC_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace