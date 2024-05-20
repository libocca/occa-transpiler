#include "attributes/attribute_names.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/handler_manager/backend_handler.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"
#include "pipeline/core/error_codes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleAtomicAttribute(SessionStage& stage, const Stmt& stmt, const Attr& attr) {
    SPDLOG_DEBUG("Handle [@atomic] attribute (stmt)");

    removeAttribute(stage, attr);
    return {};
}


__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerBackendHandler(
        TargetBackend::OPENCL, ATOMIC_ATTR_NAME, handleAtomicAttribute);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
}  // namespace
