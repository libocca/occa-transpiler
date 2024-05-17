#include "attributes/backend/metal/common.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

oklt::HandleResult handleBarrierAttribute(SessionStage& stage,
                                          const clang::Stmt& stmt,
                                          const clang::Attr& attr) {
    SPDLOG_DEBUG("Handle [@barrier] attribute");

    auto range = getAttrFullSourceRange(attr);
    stage.getRewriter().ReplaceText(range, metal::SYNC_THREADS_BARRIER);

    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok =
        registerBackendHandler(TargetBackend::METAL, BARRIER_ATTR_NAME, handleBarrierAttribute);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register {} attribute handler", BARRIER_ATTR_NAME);
    }
}
}  // namespace
