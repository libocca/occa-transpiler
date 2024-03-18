#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

namespace oklt::cuda_subset {
HandleResult handleExclusiveAttribute(const clang::Attr& attr,
                                      const clang::Decl& decl,
                                      SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif

    stage.getRewriter().RemoveText(getAttrFullSourceRange(attr));
    return defaultHandleExclusiveDeclAttribute(attr, decl, stage);
}
}  // namespace oklt::cuda_subset
