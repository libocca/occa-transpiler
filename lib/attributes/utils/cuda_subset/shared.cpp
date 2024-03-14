#include "core/attribute_manager/result.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

namespace {
const std::string SHARED_MODIFIER = "__shared__";
}
namespace oklt::cuda_subset {
HandleResult handleSharedAttribute(const clang::Attr& a, const clang::Decl& d, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    std::string replacedAttribute = " " + SHARED_MODIFIER + " ";

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
    loopInfo->shared.emplace_back(std::ref(d));

    s.getRewriter().ReplaceText(getAttrFullSourceRange(a), replacedAttribute);
    return {};
}
}  // namespace oklt::cuda_subset
