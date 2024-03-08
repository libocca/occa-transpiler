#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string restrictText = "__restrict__ ";

HandleResult handleOPENMPRestrictAttribute(const clang::Attr& a,
                                           const clang::Decl& decl,
                                           SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    auto& rewriter = s.getRewriter();

    rewriter.RemoveText(getAttrFullSourceRange(a));
    rewriter.InsertTextBefore(decl.getLocation(), restrictText);

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
