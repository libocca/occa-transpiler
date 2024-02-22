#include "attributes/attribute_names.h"
#include "core/utils/attributes.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace {
using namespace oklt;
using namespace clang;

bool handleCUDATileAttribute(const Attr* a, const Stmt* stmt, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
#endif
    return true;
}

__attribute__((constructor)) void registerBackendHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, TILE_ATTR_NAME}, AttrStmtHandler{handleCUDATileAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << OUTER_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
