#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/Sema.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling ATOMIC_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "atomic"},
    {ParsedAttr::AS_CXX11, ATOMIC_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_atomic"}};

struct AtomicAttribute : public ParsedAttrInfo {
    AtomicAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = ATOMIC_ATTRIBUTE_SPELLINGS;
        AttrKind = clang::AttributeCommonInfo::AT_Suppress;
    }

    bool diagAppertainsToStmt(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Stmt* stmt) const override {
        if (!isa<Expr, CompoundStmt>(stmt)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "expression or compound statement";
            return false;
        }
        return true;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        // INFO: fail for all decls
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "expression or compound statement";
        return false;
    }
};

ParseResult parseAtomicAttrParams(const Attr& a, SessionStage&) {
    return true;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<AtomicAttribute>(ATOMIC_ATTR_NAME,
                                                                       parseAtomicAttrParams);
}
}  // namespace
