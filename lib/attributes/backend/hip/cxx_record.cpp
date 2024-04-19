#include "attributes/utils/replace_attribute.h"
#include "core/handler_manager/implicid_handler.h"

#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclTemplate.h>

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string HIP_FUNCTION_QUALIFIER = "__device__";

HandleResult handleClassRecord(SessionStage& s, const CXXRecordDecl& d) {
    return handleCXXRecord(s, d, HIP_FUNCTION_QUALIFIER);
}

HandleResult handleClassTemplateSpecialization(SessionStage& s,
                                               const clang::ClassTemplateSpecializationDecl& d) {
    return handleCXXRecord(s, d, HIP_FUNCTION_QUALIFIER);
}

HandleResult handleClassTemplatePartialSpecialization(
    SessionStage& s,
    const clang::ClassTemplatePartialSpecializationDecl& d) {
    return handleCXXRecord(s, d, HIP_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = HandlerManager::registerImplicitHandler(TargetBackend::HIP, handleClassRecord);

    ok &= HandlerManager::registerImplicitHandler(TargetBackend::HIP, handleClassTemplateSpecialization);

    ok &= HandlerManager::registerImplicitHandler(TargetBackend::HIP, handleClassTemplatePartialSpecialization);

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register implicit handler for global function");
    }
}
}  // namespace
