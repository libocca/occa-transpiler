#include "attributes/attribute_names.h"
#include "attributes/utils/handle_restrict.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Lex/Lexer.h>

namespace {
using namespace oklt;
using namespace clang;

bool handleCUDARestrictAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
    return handleRestrictAttribute(a, d, s, "__restrict__");
}

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, RESTRICT_ATTR_NAME}, AttrDeclHandler{handleCUDARestrictAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
