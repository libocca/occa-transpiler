#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string_view SYCL_INCLUDE = "<CL/sycl.hpp>";
const std::string_view SYCL_NS = "sycl";

HandleResult handleTranslationUnitDpcpp(const clang::TranslationUnitDecl& decl, SessionStage& s) {
    return oklt::handleTranslationUnit(decl, s, {SYCL_INCLUDE}, {SYCL_NS});
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
