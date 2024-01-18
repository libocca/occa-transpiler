#include "oklt/core/attribute_names.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;

namespace oklt {

static constexpr ParsedAttrInfo::Spelling BARRIER_ATTRIBUTE_SPELLINGS[] =
    {{ParsedAttr::AS_CXX11, "barrier"},
     {ParsedAttr::AS_CXX11, BARRIER_ATTR_NAME},
     {ParsedAttr::AS_GNU, "okl_barrier"}}
;

struct BarrierAttribute : public ParsedAttrInfo {
  BarrierAttribute() {
    NumArgs = 0;
    OptArgs = 2;
    Spellings = BARRIER_ATTRIBUTE_SPELLINGS;
    AttrKind = clang::AttributeCommonInfo::AT_Suppress;
    IsStmt = true;
  }

  bool diagAppertainsToStmt(clang::Sema &sema, const clang::ParsedAttr &attr,
                            const clang::Stmt *stmt) const override {
    if (!isa<NullStmt>(stmt)) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << attr << attr.isDeclspecAttribute() << "empry statement";
      return false;
    }
    return true;
  }
};
} 

// INFO: can be moved to main
static ParsedAttrInfoRegistry::Add<oklt::BarrierAttribute>
    register_okl_barrier(oklt::BARRIER_ATTR_NAME, "");

