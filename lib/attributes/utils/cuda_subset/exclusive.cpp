#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

#include <spdlog/spdlog.h>

namespace oklt::cuda_subset {
HandleResult handleExclusiveAttribute(SessionStage& stage,
                                      const clang::Decl& decl,
                                      const clang::Attr& attr) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute");

    stage.getRewriter().RemoveText(getAttrFullSourceRange(attr));
    return defaultHandleExclusiveDeclAttribute(stage, decl, attr);
}
}  // namespace oklt::cuda_subset
