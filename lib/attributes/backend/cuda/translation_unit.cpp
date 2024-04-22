#include "attributes/utils/replace_attribute.h"

#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string CUDA_RT_INC = "<cuda_runtime.h>";
const std::string CUDA_PL_PRIM_INC = "<cuda_pipeline_primitives.h>";

std::vector<std::string_view> getBackendHeader(SessionStage& s) {
    auto defines = s.getSession().getInput().defines;
    auto hasAsyncMode = std::find(defines.begin(), defines.end(), "USE_ASYNC_READ");
    if (hasAsyncMode == defines.end()) {
        return {CUDA_RT_INC};
    }
    return {CUDA_RT_INC, CUDA_PL_PRIM_INC};
}

HandleResult handleTranslationUnit(SessionStage& s, const TranslationUnitDecl& d) {
    return handleTranslationUnit(s, d, getBackendHeader(s));
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::CUDA, clang::Decl::Kind::TranslationUnit},
        makeSpecificImplicitHandle(handleTranslationUnit));
    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register implicit handler for translation unit");
    }
}
}  // namespace
