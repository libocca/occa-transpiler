#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

bool handleOPENMPRestrictAttribute(const clang::Attr* attr,
                                   const clang::Decl* d,
                                   SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr->getNormalizedFullName() << '\n';
#endif
    auto& rewriter = stage.getRewriter();

    if (!isa<VarDecl>(d)) {
        return false;
    }
    auto varDecl = cast<VarDecl>(d);

    removeAttribute(attr, stage);

    std::string restrictText = " __restrict__ ";
    return rewriter.InsertText(varDecl->getLocation(), restrictText, false, false);
}

__attribute__((constructor)) void registerOPENMPRestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, RESTRICT_ATTR_NAME},
        AttrDeclHandler{handleOPENMPRestrictAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace