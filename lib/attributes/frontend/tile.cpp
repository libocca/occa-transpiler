#include <oklt/core/attribute_manager/attribute_manager.h>

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "oklt/core/attribute_names.h"

namespace {

using namespace oklt;
using namespace clang;

constexpr ParsedAttrInfo::Spelling TILE_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "tile"},
    {ParsedAttr::AS_CXX11, TILE_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_tile"}};

struct TileAttribute : public ParsedAttrInfo {
    TileAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = TILE_ATTRIBUTE_SPELLINGS;
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

bool parseTileAttrParams(const clang::Attr* a, SessionStage&) {
    llvm::outs() << "parse attribute: " << a->getNormalizedFullName() << '\n';
    return true;
}
__attribute__((constructor)) void registerKernelHandler() {
    AttributeManager::instance().registerAttrFrontend<TileAttribute>(TILE_ATTR_NAME,
                                                                     parseTileAttrParams);
}
}  // namespace
