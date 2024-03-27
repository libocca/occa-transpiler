#include "attributes/backend/openmp/common.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string prefixExprText = "\n#pragma omp atomic\n";
const std::string prefixCompoundText = "\n#pragma omp critical\n";

HandleResult handleOPENMPAtomicAttribute(const Attr& a, const Stmt& stmt, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    auto& rewriter = s.getRewriter();

    SourceRange attrRange = getAttrFullSourceRange(a);
    if (isa<Expr>(stmt)) {
        rewriter.ReplaceText(attrRange, prefixExprText);
        return {};
    }
    if (isa<CompoundStmt>(stmt)) {
        rewriter.ReplaceText(attrRange, prefixCompoundText);
        return {};
    }

    rewriter.RemoveText(attrRange);
    return {};
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
