#include "attributes/attribute_names.h"
#include "attributes/utils/empty_handlers.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleSharedAttribute(const Attr& a, const VarDecl& var, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] DPCPP: Handle @shared.\n";
#endif

    auto varName = var.getNameAsString();
    // Desugar since it is attributed (since it is @shared variable)
    // auto typeStr = var.getType()->getLocallyUnqualifiedSingleStepDesugaredType().getAsString();
    auto typeStr =
        QualType(var.getType().getTypePtr()->getUnqualifiedDesugaredType(), 0).getAsString();

    Error sharedError{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"};

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(sharedError);
    }
    auto* loopBelowInfo = loopInfo->getFirstAttributedChild();
    if (!loopBelowInfo || !(loopInfo->is(LoopType::Outer) && loopBelowInfo->is(LoopType::Inner))) {
        return tl::make_unexpected(sharedError);
    }

    // Save shared declaration to loopInfo
    loopInfo->shared.emplace_back(std::ref(*dyn_cast<Decl>(&var)));

    auto newDeclaration =
        util::fmt(
            "auto & {} = "
            "*(sycl::ext::oneapi::group_local_memory_for_overwrite<{}>(item_.get_group()))",
            varName,
            typeStr)
            .value();

    SourceRange range(getAttrFullSourceRange(a).getBegin(), var.getSourceRange().getEnd());

    s.getRewriter().ReplaceText(range, newDeclaration);

    return {};
}

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, SHARED_ATTR_NAME}, makeSpecificAttrHandle(handleSharedAttribute));

    // Empty Stmt hanler since @shared variable is of attributed type, it is called on DeclRefExpr
    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, SHARED_ATTR_NAME},
        makeSpecificAttrHandle(emptyHandleSharedStmtAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << SHARED_ATTR_NAME
                     << " attribute handler for DPCPP backend\n";
    }
}
}  // namespace
