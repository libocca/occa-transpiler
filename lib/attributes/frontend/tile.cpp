#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

#include "attributes/utils/parse.h"
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

constexpr int MAX_N_PARAMS = 4;

tl::expected<LoopType, Error> parseLoopType(const std::string& str) {
    if (str == "@outer") {
        return LoopType::Outer;
    } else if (str == "@inner") {
        return LoopType::Inner;
    }
    return tl::make_unexpected(Error{std::error_code(), "Tile loop type parse error"});
}

tl::expected<AttributedLoop, Error> parseLoop(const std::string& str) {
    Error err{std::error_code(), "Tile loop parse error"};

    auto nonDimLoopType = parseLoopType(str);
    // @inner or @outer (without dimensions)
    if (nonDimLoopType) {
        return AttributedLoop{nonDimLoopType.value(), Dim::X};  // x - default dimension
    }

    auto lpar_pos = str.find('(');
    auto rpar_pos = str.find(')');
    if (lpar_pos == std::string::npos || rpar_pos == std::string::npos || (lpar_pos > rpar_pos)) {
        return tl::make_unexpected(err);
    }
    auto idxNStr = str.substr(lpar_pos + 1, rpar_pos - lpar_pos - 1);
    auto loopTypeStr = str.substr(0, lpar_pos);
    llvm::outs() << "[DEBUG] idxNStr: " << idxNStr << ", loopTypeStr: " << loopTypeStr << "\n";
    auto dimIdx = util::parseStrTo<int>(idxNStr);
    auto loopType = parseLoopType(loopTypeStr);
    if (!dimIdx || !loopType) {
        return tl::make_unexpected(err);
    }
    if (dimIdx.value() < 0 || dimIdx.value() > 2) {
        return tl::make_unexpected(
            Error{std::error_code(), "AttributedLoop argument must be 0, 1, 2"});
    }
    return AttributedLoop{loopType.value(), static_cast<Dim>(dimIdx.value())};
}

tl::expected<bool, Error> parseCheck(const std::string& str) {
    constexpr const char* err_msg = "Tile check parameter format: 'check=true/false'";
    auto err = tl::make_unexpected(Error{std::error_code(), err_msg});
    auto pos = str.find("=");
    if (pos == std::string::npos || pos == (str.size() - 1)) {
        return err;
    }

    auto trueFalse = str.substr(pos + 1);
    if (trueFalse == "true") {
        return true;
    } else if (trueFalse == "false") {
        return false;
    }
    return err;
}

ParseResult parseTileAttribute(const clang::Attr* a, SessionStage& s) {
    auto tileParamsStr = parseOKLAttributeParamsStr(a);
    if (!tileParamsStr.has_value()) {
        return tl::make_unexpected(tileParamsStr.error());
    }

    auto nParams = tileParamsStr->size();
    if (nParams > MAX_N_PARAMS || nParams < 1) {
        return tl::make_unexpected(Error{{}, "@tile has 1 to 4 parameters"});
    }

    // Parse all parameters:
    // TODO: add some verification if statement evaluates to integer type
    auto tileSize = tileParamsStr.value()[0];

    std::vector<bool> checksStack;
    std::vector<AttributedLoop> loopsStack;

    // TODO: rewrite this ugly parsing if possible
    for (int i = 1; i < nParams; ++i) {
        auto currentParamStr = tileParamsStr.value()[i];
        // Parse check
        if (auto check = parseCheck(currentParamStr)) {
            checksStack.push_back(check.value());
            continue;
        }

        // TODO: dimensions should be calculated by sema if not specified
        // Parse loop
        if (auto loopType = parseLoop(currentParamStr)) {
            loopsStack.push_back(loopType.value());
            continue;
        }

        return tl::make_unexpected(Error{{}, "Can't parse tile parameter: " + currentParamStr});
    }

    // Verify number of parameters
    if (checksStack.size() > 1) {
        return tl::make_unexpected(Error{{}, "More than one tile check parameters"});
    }

    if (loopsStack.size() > 2) {
        return tl::make_unexpected(Error{{}, "More than two tile loop identifiers"});
    }

    TileParams tileParams{
        .tileSize = tileSize,
        .firstLoop = loopsStack.size() > 0 ? loopsStack[0] : AttributedLoop{},
        .secondLoop = loopsStack.size() > 1 ? loopsStack[1] : AttributedLoop{},
        .check = checksStack.size() > 0 ? checksStack.front() : true,
    };

    // Outer can't be after inner:
    if (tileParams.firstLoop.type == LoopType::Inner &&
        tileParams.secondLoop.type == LoopType::Outer) {
        return tl::make_unexpected(
            Error{{}, "Cannot have [@inner] loop outside of an [@outer] loop"});
    }

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Parsed @tile parameters: " << ": {tile size: " << tileParams.tileSize
                 << ", first loop: " << static_cast<int>(tileParams.firstLoop.type)
                 << " with dim: " << static_cast<int>(tileParams.firstLoop.dim)
                 << ", second loop: " << static_cast<int>(tileParams.secondLoop.type)
                 << " with dim: " << static_cast<int>(tileParams.secondLoop.dim)
                 << ", check: " << tileParams.check << "}\n";
#endif

    return tileParams;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<TileAttribute>(TILE_ATTR_NAME,
                                                                     parseTileAttribute);
}

}  // namespace
