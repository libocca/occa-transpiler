#include "attributes/backend/openmp/common.h"

#include <clang/Rewrite/Core/Rewriter.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string prefixText = "#pragma omp parallel for\n";

std::string getScopesCloseStr(size_t& parenCnt) {
    std::string ret;
    while (parenCnt--) {
        ret += "}\n";
    }
    return ret;
}

HandleResult handleOPENMPOuterAttribute(const Attr& a,
                                        const ForStmt& stmt,
                                        const AttributedLoop* params,
                                        SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@outer params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@outer: failed to fetch loop meta data from sema"});
    }

    auto& rewriter = s.getRewriter();

    auto parent = loopInfo->getAttributedParent();

    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.ReplaceText(attrRange, (!parent ? prefixText : ""));

    return {};
}

__attribute__((constructor)) void registerOPENMPOuterHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, OUTER_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPOuterAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << OUTER_ATTR_NAME << " attribute handler (OpenMP)\n";
    }
}
}  // namespace