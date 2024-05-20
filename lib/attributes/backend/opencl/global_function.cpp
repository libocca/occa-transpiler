#include "attributes/utils/replace_attribute.h"
#include "attributes/attribute_names.h"
#include "core/handler_manager/implicid_handler.h"
#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/var_decl.h"

#include <clang/AST/AST.h>
#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleGlobalFunction(oklt::SessionStage& s, const clang::FunctionDecl& decl) {
    if (decl.getLocation().isInvalid() || decl.isInlineBuiltinDeclaration() || !decl.hasBody()) {
        return {};
    }

    if (decl.hasAttrs()) {
        for (auto* attr : decl.getAttrs()) {
            if (attr->getNormalizedFullName() == KERNEL_ATTR_NAME) {
                SPDLOG_DEBUG(
                    "Global function handler skipped function {}, since it has @kernel attribute",
                    decl.getNameAsString());
                return {};
            }
        }
    }

    auto& r = s.getRewriter();  
    auto loc = decl.getFunctionTypeLoc();
    auto funcr = SourceRange(decl.getBeginLoc(), loc.getRParenLoc());
    auto str = r.getRewrittenText(funcr);
    
    str += ";\n";

    SPDLOG_DEBUG("Handle global function '{}' at {}",
                 decl.getNameAsString(),
                 decl.getLocation().printToString(s.getCompiler().getSourceManager()));

    r.InsertTextBefore(decl.getSourceRange().getBegin(), str);

    return {};
}

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok = registerImplicitHandler(TargetBackend::OPENCL, handleGlobalFunction);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register implicit handler for global function");
    }
}
}  // namespace
