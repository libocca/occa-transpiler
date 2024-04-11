#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string_view SYCL_INCLUDE = "<CL/sycl.hpp>";
const std::string_view SYCL_NS = "sycl";

HandleResult handleTranslationUnitDpcpp(SessionStage& s, const clang::TranslationUnitDecl& decl) {
    return oklt::handleTranslationUnit(s, decl, {SYCL_INCLUDE}, {SYCL_NS});
}

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::DPCPP, ASTNodeKind::getFromNodeKind<TranslationUnitDecl>()},
        makeSpecificImplicitHandle(handleTranslationUnitDpcpp));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register implicit handler for translation unit");
    }
}
}  // namespace
