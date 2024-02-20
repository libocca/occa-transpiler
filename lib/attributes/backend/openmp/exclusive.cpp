#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

bool handleOPENMPExclusiveAttribute(const Attr* attr, const Decl* d, SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr->getNormalizedFullName() << '\n';
#endif
    auto varDecl = dyn_cast_or_null<VarDecl>(d);
    if (!varDecl) {
        return false;
    }

    removeAttribute(attr, stage);

    // TODO: Add hasExclusive flag to currently open @outer loop.

    std::string exclusiveText = " int _occa_exclusive_index;";
    return stage.getRewriter().InsertText(varDecl->getLocation(), exclusiveText, false, true);
}

__attribute__((constructor)) void registerOPENMPExclusiveHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME},
        AttrDeclHandler{handleOPENMPExclusiveAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace