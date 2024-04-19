#include "attributes/utils/replace_attribute.h"
#include "core/handler_manager/implicid_handler.h"

#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclTemplate.h>

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string CUDA_FUNCTION_QUALIFIER = "__device__";

HandleResult handleClassRecord(SessionStage& s, const CXXRecordDecl& d) {
    return handleCXXRecord(s, d, CUDA_FUNCTION_QUALIFIER);
}

HandleResult handleClassTemplateSpecialization(SessionStage& s,
                                               const ClassTemplateSpecializationDecl& d) {
    return handleCXXRecord(s, d, CUDA_FUNCTION_QUALIFIER);
}

HandleResult handleClassTemplatePartialSpecialization(
    SessionStage& s,
    const ClassTemplatePartialSpecializationDecl& d) {
    return handleCXXRecord(s, d, CUDA_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::HandlerManager::instance().registerImplicitHandler(TargetBackend::CUDA,
                                                                       handleClassRecord);

    ok &= oklt::HandlerManager::instance().registerImplicitHandler(
        TargetBackend::CUDA, handleClassTemplateSpecialization);

    ok &= oklt::HandlerManager::instance().registerImplicitHandler(
        TargetBackend::CUDA, handleClassTemplatePartialSpecialization);

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register implicit handler for global function");
    }
}
}  // namespace
