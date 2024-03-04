#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

std::string prefixText = "#pragma omp parallel for\n";

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

    auto& astCtx = s.getCompiler().getASTContext();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@outer: failed to fetch loop meta data from sema"});
    }

    // Traverse up till Kernel or attributed loop.
    auto parent = loopInfo->parent;
    while (parent && parent->metadata.type == LoopMetaType::Regular) {
        parent = parent->parent;
    }

    SourceRange attr_range = getAttrFullSourceRange(a);
    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1)
        .addReplacement(OKL_TRANSPILED_ATTR, attr_range, (!parent ? prefixText : ""))
        .build();
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