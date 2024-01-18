#include "oklt/core/attribute_names.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Basic/DiagnosticSema.h"

using namespace clang;

namespace oklt {

static constexpr ParsedAttrInfo::Spelling INNER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "inner"},
    {ParsedAttr::AS_CXX11, INNER_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_inner"}};

struct InnerAttribute : public ParsedAttrInfo {
  InnerAttribute() {
    NumArgs = 0;
    OptArgs = 2;
    Spellings = INNER_ATTRIBUTE_SPELLINGS;
    AttrKind = clang::AttributeCommonInfo::AT_Suppress;
    IsStmt = true;
  }

  bool diagAppertainsToStmt(clang::Sema &sema, const clang::ParsedAttr &attr,
                            const clang::Stmt *stmt) const override {
    if (!isa<ForStmt>(stmt)) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << attr << attr.isDeclspecAttribute() << "for statement";
      return false;
    }
    return true;
  }
};
} 

// INFO: can be moved to main
static ParsedAttrInfoRegistry::Add<oklt::InnerAttribute> register_okl_inner(oklt::INNER_ATTR_NAME,
                                                                           "");

