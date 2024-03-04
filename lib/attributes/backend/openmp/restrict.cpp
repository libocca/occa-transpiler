#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOPENMPRestrictAttribute(const clang::Attr& attr,
                                           const clang::VarDecl& decl,
                                   SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif

    removeAttribute(attr, stage);

    std::string restrictText = " __restrict__ ";
    stage.getRewriter().InsertText(decl.getLocation(), restrictText, false, false);

    return {};
}

__attribute__((constructor)) void registerOPENMPRestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, RESTRICT_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPRestrictAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace