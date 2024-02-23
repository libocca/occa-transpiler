#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling SHARED_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "shared"},
    {ParsedAttr::AS_CXX11, SHARED_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_shared"}};

struct SharedAttribute : public ParsedAttrInfo {
    SharedAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = SHARED_ATTRIBUTE_SPELLINGS;
        AttrKind = clang::AttributeCommonInfo::AT_Annotate;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        // INFO: this attribute appertains to functions only.
        if (!isa<VarDecl, TypeDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "variable or type declaration";
            return false;
        }

        // INFO: if VarDecl, check if array
        if (auto* var_decl = dyn_cast_or_null<VarDecl>(decl)) {
            if (!dyn_cast_or_null<ConstantArrayType>(var_decl->getType().getTypePtr())) {
                sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                    << attr << attr.isDeclspecAttribute() << "array declaration";
            }
        }
        return true;
    }

    bool diagAppertainsToStmt(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Stmt* stmt) const override {
        // INFO: fail for all statements
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "variable or type declaration";
        return false;
    }
};

tl::expected<std::any, Error> parseSharedAttrParams(const clang::Attr* a, SessionStage&) {
    return true;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<SharedAttribute>(SHARED_ATTR_NAME,
                                                                       parseSharedAttrParams);
}
}  // namespace
