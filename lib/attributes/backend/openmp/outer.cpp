#include "attributes/attribute_names.h"
#include "attributes/backend/openmp/common.h"
#include "attributes/frontend/params/loop.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

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

    auto trans =
        TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1);

    auto parent = loopInfo->getAttributedParent();

    SourceRange attrRange = getAttrFullSourceRange(a);
    trans.addReplacement(OKL_TRANSPILED_ATTR, attrRange, (!parent ? prefixText : ""));

    // process `@exclusive` that are within current loop.
    if (auto procExclusive = openmp::postHandleExclusive(*loopInfo, trans);
        !procExclusive.has_value()) {
        return procExclusive;
    }

    // process `@shared` that are within current loop.
    if (auto procShared = openmp::postHandleShared(*loopInfo, trans); !procShared.has_value()) {
        return procShared;
    }

    return trans.build();
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