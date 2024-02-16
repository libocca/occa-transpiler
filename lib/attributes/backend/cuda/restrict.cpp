#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/utils/attributes.h>

namespace {
using namespace oklt;
using namespace clang;

// bool parseRestrictAttribute(const clang::Attr* a, SessionStage&) {
// #ifdef TRANSPILER_DEBUG_LOG
//   llvm::outs() << "parse attribute: " << a->getNormalizedFullName() << '\n';
// #endif
//   return true;
// }

bool handleRestrictAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
#endif
    auto& rewriter = s.getRewriter();

    if (!isa<VarDecl>(d)) {
        return false;
    }

    auto varDecl = cast<VarDecl>(d);
    SourceLocation identifierLoc = varDecl->getLocation();
    removeAttribute(a, s);
    std::string restrictText = " __restrict__ ";
    return rewriter.InsertText(identifierLoc, restrictText, false, false);
}

__attribute__((constructor)) void registerRestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, RESTRICT_ATTR_NAME}, AttrDeclHandler{handleRestrictAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
