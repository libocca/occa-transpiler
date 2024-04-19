#include "attributes/frontend/params/tile.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"

#include "core/handler_manager/parse_handler.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {
using namespace oklt;
using namespace clang;

constexpr ParsedAttrInfo::Spelling TILE_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, TILE_ATTR_NAME},
    {ParsedAttr::AS_GNU, TILE_ATTR_NAME}};

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

HandleResult parseTileAttribute(SessionStage& stage, const clang::Attr& attr, OKLParsedAttr& data) {
    TileParams ret = {};
    if (data.args.empty()) {
        return tl::make_unexpected(Error{{}, "[@tile] expects at least one argument"});
    }

    if (data.args.size() > 3) {
        return tl::make_unexpected(
            Error{{},
                  "[@tile] takes 1-3 arguments, the last 2 being attributes for the block "
                  "and in-block loops respectively"});
    }

    if (data.args[0].empty()) {
        return tl::make_unexpected(Error{{}, "[@tile] expects a non-empty first argument"});
    }
    ret.tileSize = data.args[0].getRaw();

    for (size_t i = 1; i < data.args.size(); ++i) {
        if (!data.isa<OKLParsedAttr>(i)) {
            return tl::make_unexpected(
                Error{{}, "[@tile] can only take attributes for the 2nd and 3rd arguments"});
        }

        auto subAttr = data.get<OKLParsedAttr>(i).value();
        auto loop = stage.getAttrManager().parseAttr(stage, attr, subAttr);
        if (!loop) {
            return tl::make_unexpected(loop.error());
        }

        if (loop.value().type() != typeid(AttributedLoop)) {
            return tl::make_unexpected(Error{{}, "[@tile] loop type parse error"});
        }

        if (i == 1) {
            ret.firstLoop = std::any_cast<AttributedLoop>(loop.value());
            continue;
        }
        if (i == 2) {
            ret.secondLoop = std::any_cast<AttributedLoop>(loop.value());
            continue;
        }
    }

    for (const auto& param : data.kwargs) {
        if (param.first != "check") {
            return tl::make_unexpected(Error{{}, "[@tile] does not take this kwarg"});
        }

        if (!param.second.isa<bool>()) {
            return tl::make_unexpected(Error{{}, "[@tile] 'check' argument must be true or false"});
        }
        param.second.getTo(ret.check);
    }

    return ret;
}

__attribute__((constructor)) void registerAttrFrontend() {
    HandlerManager::instance().registerAttrFrontend<TileAttribute>(TILE_ATTR_NAME,
                                                                   parseTileAttribute);
}

}  // namespace
