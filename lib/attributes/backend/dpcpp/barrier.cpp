#include "attributes/attribute_names.h"
#include "attributes/backend/dpcpp/common.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleBarrierAttribute(SessionStage& s,
                                    const clang::Stmt& stmt,
                                    const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@barrier] attribute");

    SourceRange range(getAttrFullSourceRange(a).getBegin(), stmt.getEndLoc());
    s.getRewriter().ReplaceText(range, dpcpp::SYNC_THREADS_BARRIER);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, BARRIER_ATTR_NAME, ASTNodeKind::getFromNodeKind<Stmt>()},
        makeSpecificAttrHandle(handleBarrierAttribute));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", BARRIER_ATTR_NAME);
    }
}
}  // namespace
