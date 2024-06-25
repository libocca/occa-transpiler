#include "attributes/attribute_names.h"
#include "attributes/backend/opencl/common.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/handler_manager/backend_handler.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string BARRIER_STR = "barrier(CLK_LOCAL_MEM_FENCE);\n";

HandleResult handleBarrierAttribute(SessionStage& s,
                                    const clang::Stmt& stmt,
                                    const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@barrier] attribute");

    SourceRange range(getAttrFullSourceRange(a).getBegin(), stmt.getEndLoc());
    s.getRewriter().ReplaceText(range, BARRIER_STR);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok =
        registerBackendHandler(TargetBackend::OPENCL, BARRIER_ATTR_NAME, handleBarrierAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register {} attribute handler", BARRIER_ATTR_NAME);
    }
}
}  // namespace
