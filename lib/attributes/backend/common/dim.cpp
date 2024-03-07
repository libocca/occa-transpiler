#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

HandleResult handleDimDeclAttribute(const clang::Attr& a,
                                   const clang::Decl& decl,
                                   const std::any* params,
                                   SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle decl attribute: " << a.getNormalizedFullName() << '\n';
#endif
    return {};
}

HandleResult handleDimStmtAttribute(const clang::Attr& a,
                                   const clang::Stmt& stmt,
                                   const std::any* params,
                                   SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle stmt attribute: " << a.getNormalizedFullName() << '\n';
#endif
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIM_ATTR_NAME, AttrDeclHandler{handleDimDeclAttribute});
    if (!ok) {
        llvm::errs() << "failed to register " << DIM_ATTR_NAME << " attribute decl handler\n";
    }

    ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIM_ATTR_NAME, AttrStmtHandler{handleDimStmtAttribute});
    if (!ok) {
        llvm::errs() << "failed to register " << DIM_ATTR_NAME << " attribute stmt handler\n";
    }
}
}  // namespace
