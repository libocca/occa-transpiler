#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/default_handlers.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;
HandleResult handleExclusiveAttribute(SessionStage& s,
                                      const clang::Decl& decl,
                                      const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute");
    s.getRewriter().RemoveText(getAttrFullSourceRange(a));
    return defaultHandleExclusiveDeclAttribute(s, decl, a);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(handleExclusiveAttribute));

    ok &= oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(defaultHandleExclusiveStmtAttribute));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
