#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>

namespace {
using namespace oklt;
using namespace clang;
HandleResult handleExclusiveAttribute(const clang::Attr& a,
                                      const clang::Decl& stmt,
                                      SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] handle attribute: @exclusive\n";
#endif
    auto& rewriter = s.getRewriter();
    SourceRange range(a.getRange().getBegin().getLocWithOffset(-2),
                      a.getRange().getEnd().getLocWithOffset(2));
    rewriter.RemoveText(range);

    return true;
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(handleExclusiveAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME
                     << " attribute handler (DPCPP)\n";
    }
}
}  // namespace
