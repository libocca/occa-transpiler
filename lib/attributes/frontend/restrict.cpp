#include "oklt/core/attribute_names.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/DiagnosticSema.h"

using namespace clang;

namespace oklt {

static constexpr ParsedAttrInfo::Spelling RESTRICT_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "restrict"},
    {ParsedAttr::AS_CXX11, RESTRICT_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_restrict"}};

struct RestrictAttribute : public ParsedAttrInfo {
  RestrictAttribute() {
    NumArgs = 0;
    OptArgs = 0;
    Spellings = RESTRICT_ATTRIBUTE_SPELLINGS;
  }

  bool diagAppertainsToDecl(clang::Sema &sema, const clang::ParsedAttr &attr,
                            const clang::Decl *decl) const override {
    // INFO: this can be as the function argument
    if (!isa<ParmVarDecl>(decl)) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << attr << ": can be applied only for parameters of pointer type in function";
      return false;
    }
    const auto *param = cast<ParmVarDecl>(decl);
    if (!param->getType()->isPointerType()) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << attr << ": supports only pointer type";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema &s, Decl *d, const ParsedAttr &a) const override {
    d->addAttr(AnnotateAttr::Create(s.Context, oklt::RESTRICT_ATTR_NAME, nullptr, 0, a));

    return AttrHandling::AttributeApplied;
  }
};
} 

// INFO: can be moved to main
static ParsedAttrInfoRegistry::Add<oklt::RestrictAttribute>
    register_okl_resitrct(oklt::RESTRICT_ATTR_NAME, "");

