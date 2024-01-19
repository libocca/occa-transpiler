#include "oklt/core/attribute_names.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/DiagnosticSema.h"

using namespace clang;

namespace oklt {
static constexpr ParsedAttrInfo::Spelling KERNEL_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, KERNEL_ATTR_NAME},
    {ParsedAttr::AS_CXX11, "kernel"},
    {ParsedAttr::AS_GNU, "okl_kernel"}};

struct KernelAttribute : public ParsedAttrInfo {
  KernelAttribute() {
    NumArgs = 1;
    OptArgs = 1;
    Spellings = KERNEL_ATTRIBUTE_SPELLINGS;
    AttrKind = clang::AttributeCommonInfo::AT_Annotate;
  }

  bool diagAppertainsToDecl(clang::Sema &sema, const clang::ParsedAttr &attr,
                            const clang::Decl *decl) const override {
    // INFO: this decl function can be saved to global map.
    //       in this case there is no need to make attribute !!!
    // INFO: this attribute appertains to functions only.
    if (!isa<FunctionDecl>(decl)) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << attr << attr.isDeclspecAttribute() << "functions";
      return false;
    }
    return true;
  }
};
} // namespace okl

// INFO: can be moved to main
static ParsedAttrInfoRegistry::Add<oklt::KernelAttribute> register_okl_kernel(oklt::KERNEL_ATTR_NAME,
                                                                             "");
