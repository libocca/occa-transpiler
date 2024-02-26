#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/Sema.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling BARRIER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "barrier"},
    {ParsedAttr::AS_CXX11, BARRIER_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_barrier"}};

struct BarrierAttribute : public ParsedAttrInfo {
    BarrierAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = BARRIER_ATTRIBUTE_SPELLINGS;
        AttrKind = clang::AttributeCommonInfo::AT_Suppress;
        IsStmt = true;
    }

    bool diagAppertainsToStmt(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Stmt* stmt) const override {
        if (!isa<NullStmt>(stmt)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "empty statement";
            return false;
        }
        return true;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        // INFO: fail for all decls
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "empty statements";
        return false;
    }
};

ParseResult parseBarrierAttrParams(const clang::Attr* a, SessionStage&) {
    return true;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<BarrierAttribute>(BARRIER_ATTR_NAME,
                                                                        parseBarrierAttrParams);
}
}  // namespace
