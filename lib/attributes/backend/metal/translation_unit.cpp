#include "attributes/utils/replace_attribute.h"
#include "core/handler_manager/implicid_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string_view METAL_INCLUDE1 = "<metal_compute>";
const std::string_view METAL_INCLUDE2 = "<metal_stdlib>";
const std::string_view METAL_NS = "metal";

HandleResult handleTranslationUnit(SessionStage& s, const clang::TranslationUnitDecl& decl) {
    return oklt::handleTranslationUnit(s, decl, {METAL_INCLUDE1, METAL_INCLUDE2}, {METAL_NS});
}

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok = registerImplicitHandler(TargetBackend::METAL, handleTranslationUnit);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register implicit handler for translation unit");
    }
}
}  // namespace
