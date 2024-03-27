#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleTranslationUnitDpcpp(const clang::TranslationUnitDecl& decl, SessionStage& s) {
    const std::string SYCL_INCLUDE = "<CL/sycl.hpp>\nusing namespace sycl;\n";
    return oklt::handleTranslationUnit(decl, s, SYCL_INCLUDE);
}

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::DPCPP, clang::Decl::Kind::TranslationUnit},
        makeSpecificImplicitHandle(handleTranslationUnitDpcpp));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register implicit handler for translation unit");
    }
}
}  // namespace
