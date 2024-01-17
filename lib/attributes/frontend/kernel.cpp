#include "oklt/core/attribute_names.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
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
    NumArgs = 0;
    OptArgs = 1;
    Spellings = KERNEL_ATTRIBUTE_SPELLINGS;
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

  AttrHandling handleDeclAttribute(Sema &s, Decl *d, const ParsedAttr &a) const override {
    _attr_args.resize(a.getNumArgs());
    for (int i = 0; i < a.getNumArgs(); i++) {
      _attr_args[i] = a.getArgAsExpr(i);
    }

    d->addAttr(AnnotateAttr::Create(s.Context, oklt::KERNEL_ATTR_NAME, _attr_args.data(),
                                    _attr_args.size(), a));

    return AttrHandling::AttributeApplied;
  }

  std::vector<Expr *> mutable _attr_args;
};
} // namespace okl

// INFO: can be moved to main
static ParsedAttrInfoRegistry::Add<oklt::KernelAttribute> register_okl_kernel(oklt::KERNEL_ATTR_NAME,
                                                                             "");
