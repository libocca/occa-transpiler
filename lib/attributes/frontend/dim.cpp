#include "oklt/core/attribute_names.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/DiagnosticSema.h"

using namespace clang;

namespace oklt {

static constexpr ParsedAttrInfo::Spelling DIM_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "dim"},
    {ParsedAttr::AS_CXX11, DIM_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_dim"}};

struct DimAttribute : public ParsedAttrInfo {
  DimAttribute() {
    NumArgs = 1;
    OptArgs = 6;
    Spellings = DIM_ATTRIBUTE_SPELLINGS;
    AttrKind = clang::AttributeCommonInfo::AT_Annotate;
  }

  bool diagAppertainsToDecl(clang::Sema &sema, const clang::ParsedAttr &attr,
                            const clang::Decl *decl) const override {
    // INFO: this decl function can be saved to global map.
    //       in this case there is no need to make attribute !!!
    // INFO: this attribute appertains to functions only.

    if (!isa<VarDecl>(decl) && !isa<TypeDecl>(decl) && !isa<TypedefDecl>(decl) &&
        !isa<TypedefNameDecl>(decl)) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << attr << attr.isDeclspecAttribute() << "functions";
      return false;
    }
    return true;
  }
};
}

static ParsedAttrInfoRegistry::Add<oklt::DimAttribute> register_okl_sim(oklt::DIM_ATTR_NAME, "");

