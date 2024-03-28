#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

#include <clang/AST/DeclCXX.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

const std::string HIP_FUNCTION_QUALIFIER = "__device__";

HandleResult handleHIPCXXRecord(const clang::CXXRecordDecl& d, SessionStage& s) {
    return handleCXXRecord(d, s, HIP_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, clang::Decl::Kind::CXXRecord},
        makeSpecificImplicitHandle(handleHIPCXXRecord));

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register implicit handler for global function");
    }
}
}  // namespace
