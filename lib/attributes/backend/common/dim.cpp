#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

tl::expected<std::any, Error> handleDimDeclAttrbute(const clang::Attr* a,
                                                    const clang::Decl* decl,
                                                    const std::any& params,
                                                    SessionStage& s) {
    llvm::outs() << "handle decl attribute: " << a->getNormalizedFullName() << '\n';
    return true;
}

tl::expected<std::any, Error> handleDimStmtAttrbute(const clang::Attr* a,
                                                    const clang::Stmt* stmt,
                                                    const std::any& params,
                                                    SessionStage& s) {
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
