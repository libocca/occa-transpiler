#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOPENMPExclusiveAttribute(const Attr& attr,
                                            const VarDecl& var,
                                            SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif
    removeAttribute(attr, stage);

    // TODO: Add hasExclusive flag to currently open @outer loop.

    std::string exclusiveText = " int _occa_exclusive_index;";
    stage.getRewriter().InsertText(var.getLocation(), exclusiveText, false, true);

    return {};
}

__attribute__((constructor)) void registerOPENMPExclusiveHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPExclusiveAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace