#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

bool handleDimDeclAttrbute(const clang::Attr* a, const clang::Decl* decl, SessionStage& s) {
    llvm::outs() << "handle decl attribute: " << a->getNormalizedFullName() << '\n';
    return true;
}

bool handleDimStmtAttrbute(const clang::Attr* a, const clang::Stmt* stmt, SessionStage& s) {
    llvm::outs() << "handle stmt attribute: " << a->getNormalizedFullName() << '\n';
    return true;
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIM_ATTR_NAME, AttrDeclHandler{handleDimDeclAttrbute});
    if (!ok) {
        llvm::errs() << "failed to register " << DIM_ATTR_NAME << " attribute decl handler\n";
    }

    ok = oklt::AttributeManager::instance().registerCommonHandler(
        DIM_ATTR_NAME, AttrStmtHandler{handleDimStmtAttrbute});
    if (!ok) {
        llvm::errs() << "failed to register " << DIM_ATTR_NAME << " attribute stmt handler\n";
    }
}
}  // namespace
