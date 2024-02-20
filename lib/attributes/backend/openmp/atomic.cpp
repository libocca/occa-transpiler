#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

bool handleOPENMPAtomicAttribute(const Attr* attr, const Decl* d, SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr->getNormalizedFullName() << '\n';
#endif

    removeAttribute(attr, stage);

    return true;
}

__attribute__((constructor)) void registerOPENMPAtomicHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, ATOMIC_ATTR_NAME}, AttrDeclHandler{handleOPENMPAtomicAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << ATOMIC_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace