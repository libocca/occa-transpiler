#include "attributes/utils/replace_attribute.h"
#include "core/handler_manager/implicid_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string_view OPENCL_PRAGMA = "#pragma OPENCL EXTENSON cl_khr_fp64 : enable";

HandleResult handleTranslationUnitOpencl(SessionStage& s, const clang::TranslationUnitDecl& decl) {
    return oklt::handleTranslationUnit(s, decl, {}, {OPENCL_PRAGMA}, {});
}

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok = registerImplicitHandler(TargetBackend::OPENCL, handleTranslationUnitOpencl);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register implicit handler for translation unit");
    }
}
}  // namespace
