#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "params/tile.h"

#include <oklt/util/string_utils.h>

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

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

tl::expected<AttributedLoop, Error> parseOuterLoop(OKLParsedAttr& attrData) {
    if (!attrData.kwargs.empty()) {
        return tl::make_unexpected(Error{std::error_code(), "[@outer] does not take kwargs"});
    }

    if (attrData.args.size() > 1) {
        return tl::make_unexpected(Error{std::error_code(), "[@outer] takes at most one index"});
    }

    AttributedLoop ret{
        .type = LoopType::Outer,
        .dim = Dim::Auto,
    };

    if (auto dimSize = attrData.get<int>(0); dimSize.has_value()) {
        if (dimSize.value() < 0 || dimSize.value() > 2) {
            return tl::make_unexpected(
                Error{std::error_code(), "[@outer] argument must be 0, 1, or 2"});
        }
        ret.dim = static_cast<Dim>(dimSize.value());
    }

    return ret;
};

tl::expected<AttributedLoop, Error> parseInnerLoop(OKLParsedAttr& attrData) {
    if (!attrData.kwargs.empty()) {
        return tl::make_unexpected(Error{std::error_code(), "[@inner] does not take kwargs"});
    }

    if (attrData.args.size() > 1) {
        return tl::make_unexpected(Error{std::error_code(), "[@inner] takes at most one index"});
    }

    AttributedLoop ret{
        .type = LoopType::Outer,
        .dim = Dim::Auto,
    };

    if (auto dimSize = attrData.get<int>(0); dimSize.has_value()) {
        if (dimSize.value() < 0 || dimSize.value() > 2) {
            return tl::make_unexpected(
                Error{std::error_code(), "[@inner] argument must be 0, 1, or 2"});
        }
        ret.dim = static_cast<Dim>(dimSize.value());
    }

    return ret;
};

tl::expected<AttributedLoop, Error> parseLoopType(OKLParsedAttr& attrData) {
    if (attrData.name == OUTER_ATTR_NAME) {
        return parseOuterLoop(attrData);
    } else if (attrData.name == INNER_ATTR_NAME) {
        return parseInnerLoop(attrData);
    }
    return tl::make_unexpected(Error{std::error_code(), "[@tile] loop type parse error"});
};

ParseResult parseTileAttribute(const clang::Attr& attr, OKLParsedAttr& data, SessionStage& stage) {
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

    for (auto i = size_t{1}; i < data.args.size(); ++i) {
        if (!data.isa<OKLParsedAttr>(i)) {
            return tl::make_unexpected(
                Error{{}, "[@tile] can only take attributes for the 2nd and 3rd arguments"});
        }

        auto subAttr = data.get<OKLParsedAttr>(i).value();
        auto loop = parseLoopType(subAttr);
        if (!loop) {
            return tl::make_unexpected(loop.error());
        }

        if (i == 1) {
            ret.firstLoop = loop.value();
            continue;
        }
        if (i == 2) {
            ret.secondLoop = loop.value();
            continue;
        }
    }

    for (auto param : data.kwargs) {
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
    AttributeManager::instance().registerAttrFrontend<TileAttribute>(TILE_ATTR_NAME,
                                                                     parseTileAttribute);
}

}  // namespace
