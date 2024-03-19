#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>

namespace {
using namespace oklt;
using namespace clang;
HandleResult handleExclusiveAttribute(const clang::Attr& a,
                                      const clang::Decl& decl,
                                      SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] handle attribute: @exclusive\n";
#endif
    s.getRewriter().RemoveText(getAttrFullSourceRange(a));
    return defaultHandleExclusiveDeclAttribute(a, decl, s);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(handleExclusiveAttribute));

    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(defaultHandleExclusiveStmtAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME
                     << " attribute handler (DPCPP)\n";
    }
}
}  // namespace
