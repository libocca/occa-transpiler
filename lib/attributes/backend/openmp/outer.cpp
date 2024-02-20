#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

bool handleOPENMPOuterAttribute(const Attr* attr, const Stmt* s, SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr->getNormalizedFullName() << '\n';
#endif
    auto forStmt = dyn_cast_or_null<ForStmt>(s);
    if (!forStmt) {
        return false;
    }

    removeAttribute(attr, stage);

    std::string outerText = "#pragma omp parallel for\n";
    return stage.getRewriter().InsertText(forStmt->getBeginLoc(), outerText, false, true);
}

__attribute__((constructor)) void registerOPENMPOuterHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, EXCLUSIVE_ATTR_NAME}, AttrStmtHandler{handleOPENMPOuterAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace