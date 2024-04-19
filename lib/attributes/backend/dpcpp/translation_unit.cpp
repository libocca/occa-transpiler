#include "attributes/utils/replace_attribute.h"
#include "core/handler_manager/implicid_handler.h"

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
    auto ok = oklt::HandlerManager::instance().registerImplicitHandler(TargetBackend::DPCPP, handleTranslationUnitDpcpp);

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register implicit handler for translation unit");
    }
}
}  // namespace
