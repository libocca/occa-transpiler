#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleMaxInnerDimsStmtAttribute(const clang::Attr& a,
                                             const clang::ForStmt&,
                                             const AttributedLoopInnerSize* params,
                                             SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    removeAttribute(a, s);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        MAX_INNER_DIMS, makeSpecificAttrHandle(handleMaxInnerDimsStmtAttribute));
    if (!ok) {
        llvm::errs() << "failed to register " << MAX_INNER_DIMS << " attribute decl handler\n";
    }
}
}  // namespace
