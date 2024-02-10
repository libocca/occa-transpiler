#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "oklt/core/attribute_names.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling INNER_ATTRIBUTE_SPELLINGS[] = {
  {ParsedAttr::AS_CXX11, "inner"},
  {ParsedAttr::AS_CXX11, INNER_ATTR_NAME},
  {ParsedAttr::AS_GNU, "okl_inner"}};

struct InnerAttribute : public ParsedAttrInfo {
  InnerAttribute() {
    NumArgs = 1;
    OptArgs = 0;
    Spellings = INNER_ATTRIBUTE_SPELLINGS;
    AttrKind = clang::AttributeCommonInfo::AT_Suppress;
    IsStmt = true;
  }

  bool diagAppertainsToStmt(clang::Sema& sema,
                            const clang::ParsedAttr& attr,
                            const clang::Stmt* stmt) const override {
    if (!isa<ForStmt>(stmt)) {
      sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
        << attr << attr.isDeclspecAttribute() << "for statement";
      return false;
    }
    return true;
  }

  bool diagAppertainsToDecl(clang::Sema& sema,
                            const clang::ParsedAttr& attr,
                            const clang::Decl* decl) const override {
    // INFO: fail for all decls
    sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
      << attr << attr.isDeclspecAttribute() << "for statement";
    return false;
  }
};

ParsedAttrInfoRegistry::Add<InnerAttribute> register_okl_inner(INNER_ATTR_NAME, "");
}  // namespace
