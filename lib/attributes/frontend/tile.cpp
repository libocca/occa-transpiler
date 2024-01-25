#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "oklt/core/attribute_names.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling TILE_ATTRIBUTE_SPELLINGS[] = {
  {ParsedAttr::AS_CXX11, "tile"},
  {ParsedAttr::AS_CXX11, TILE_ATTR_NAME},
  {ParsedAttr::AS_GNU, "okl_tile"}};

struct TileAttribute : public ParsedAttrInfo {
  TileAttribute() {
    NumArgs = 1;
    OptArgs = 2;
    Spellings = TILE_ATTRIBUTE_SPELLINGS;
    AttrKind = clang::AttributeCommonInfo::AT_Suppress;
    IsStmt = true;
  }
  bool diagAppertainsToDecl(clang::Sema& sema,
                            const clang::ParsedAttr& attr,
                            const clang::Decl* decl) const override {
    if (decl->getFriendObjectKind()) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << attr << attr.isDeclspecAttribute() << "for statement";
      return false;
    }
    return true;
  }
};

ParsedAttrInfoRegistry::Add<TileAttribute> register_okl_tile(TILE_ATTR_NAME, "");
}  // namespace
