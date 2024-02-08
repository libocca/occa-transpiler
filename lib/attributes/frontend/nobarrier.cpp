#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "oklt/core/attribute_names.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling NOBARRIER_ATTRIBUTE_SPELLINGS[] = {
  {ParsedAttr::AS_CXX11, "nobarrier"},
  {ParsedAttr::AS_CXX11, NOBARRIER_ATTR_NAME},
  {ParsedAttr::AS_GNU, "okl_nobarrier"}};

struct NoBarrierAttribute : public ParsedAttrInfo {
  NoBarrierAttribute() {
    NumArgs = 1;
    OptArgs = 0;
    Spellings = NOBARRIER_ATTRIBUTE_SPELLINGS;
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

ParsedAttrInfoRegistry::Add<NoBarrierAttribute> register_okl_nobarrier(NOBARRIER_ATTR_NAME, "");
}  // namespace