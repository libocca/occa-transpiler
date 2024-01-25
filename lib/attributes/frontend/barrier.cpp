#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "oklt/core/attribute_names.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling BARRIER_ATTRIBUTE_SPELLINGS[] = {
  {ParsedAttr::AS_CXX11, "barrier"},
  {ParsedAttr::AS_CXX11, BARRIER_ATTR_NAME},
  {ParsedAttr::AS_GNU, "okl_barrier"}};

struct BarrierAttribute : public ParsedAttrInfo {
  BarrierAttribute() {
    NumArgs = 1;
    OptArgs = 2;
    Spellings = BARRIER_ATTRIBUTE_SPELLINGS;
    AttrKind = clang::AttributeCommonInfo::AT_Suppress;
    IsStmt = true;
  }

  bool diagAppertainsToStmt(clang::Sema& sema,
                            const clang::ParsedAttr& attr,
                            const clang::Stmt* stmt) const override {
    if (!isa<NullStmt>(stmt)) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << attr << attr.isDeclspecAttribute() << "empry statement";
      return false;
    }
    return true;
  }
};

ParsedAttrInfoRegistry::Add<BarrierAttribute> register_okl_barrier(BARRIER_ATTR_NAME, "");
}  // namespace

