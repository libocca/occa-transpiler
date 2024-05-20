#include "attributes/utils/replace_attribute.h"

#include "core/handler_manager/implicid_handler.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string CUDA_RT_INC = "<cuda_runtime.h>";

std::vector<std::string_view> getBackendHeader(SessionStage& s) {
    return {CUDA_RT_INC};
}

HandleResult handleTranslationUnit(SessionStage& s, const TranslationUnitDecl& d) {
    return handleTranslationUnit(s, d, getBackendHeader(s), {});
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerImplicitHandler(TargetBackend::CUDA, handleTranslationUnit);
    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register implicit handler for translation unit");
    }
}
}  // namespace
