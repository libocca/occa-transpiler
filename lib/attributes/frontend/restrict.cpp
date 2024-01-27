#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "oklt/core/attribute_names.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling RESTRICT_ATTRIBUTE_SPELLINGS[] = {
  {ParsedAttr::AS_CXX11, "restrict"},
  {ParsedAttr::AS_CXX11, RESTRICT_ATTR_NAME},
  {ParsedAttr::AS_GNU, "okl_restrict"}};

struct RestrictAttribute : public ParsedAttrInfo {
  RestrictAttribute() {
    NumArgs = 1;
    OptArgs = 0;
    Spellings = RESTRICT_ATTRIBUTE_SPELLINGS;
    AttrKind = clang::AttributeCommonInfo::AT_Annotate;
  }

  bool diagAppertainsToDecl(clang::Sema& sema,
                            const clang::ParsedAttr& attr,
                            const clang::Decl* decl) const override {
    // INFO: this can be as the function argument
    if (!isa<ParmVarDecl>(decl)) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << attr << ": can be applied only for parameters of pointer type in function";
      return false;
    }
    const auto* param = cast<ParmVarDecl>(decl);
    if (!param->getType()->isPointerType()) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << attr << ": supports only pointer type";
      return false;
    }
    return true;
  }
};

ParsedAttrInfoRegistry::Add<RestrictAttribute> register_okl_resitrct(RESTRICT_ATTR_NAME, "");
}  // namespace
