#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace oklt {
class SessionStage;
}

namespace {
using namespace oklt;
using namespace clang;

const std::string HIP_RT_INC = "<hip/hip_runtime.h>";
HandleResult handleTU(SessionStage& s, const TranslationUnitDecl& d) {
    return handleTranslationUnit(s, d, {HIP_RT_INC});
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, ASTNodeKind::getFromNodeKind<TranslationUnitDecl>()},
        makeSpecificImplicitHandle(handleTU));

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register implicit handler for translation unit");
    }
}
}  // namespace
