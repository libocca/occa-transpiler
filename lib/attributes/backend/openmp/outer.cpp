#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/ParentMapContext.h>

namespace {
using namespace oklt;
using namespace clang;

bool isRootLevel(ParentMapContext& ctx, OklSemaCtx& sema, const DynTypedNode node) {
    auto parents = ctx.getParents(node);
    while (!parents.empty()) {
        if (auto funcStmt = parents[0].get<FunctionDecl>()) {
            return true;
        }

        if (auto forStmt = parents[0].get<ForStmt>()) {
            auto metadata = sema.getLoopMetaData(*forStmt);
            if (metadata.has_value()) {
                return false;
            }
        };

        parents = ctx.getParents(parents[0]);
    }

    return true;
}

bool handleOPENMPOuterAttribute(const Attr& attr,
                                const ForStmt& stmt,
                                const AttributedLoop* params,
                                SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif
    removeAttribute(attr, stage);

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto forLoopMetaData = sema.getLoopMetaData(stmt);
    if (!forLoopMetaData) {
        stage.pushError(std::error_code(), "@tile: failed to fetch loop meta data from sema");
        return false;
    }

    auto& ctx = stage.getCompiler().getASTContext();
    if (!isRootLevel(ctx.getParentMapContext(), sema, DynTypedNode::create(stmt))) {
        return true;
    }

    std::string outerText = "#pragma omp parallel for\n";
    return stage.getRewriter().InsertText(stmt.getBeginLoc(), outerText, false, true);
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