#include <clang/AST/Decl.h>
#include <oklt/attributes/frontend/parsers/tile.h>
#include <oklt/attributes/frontend/utils/parse.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/util/string_utils.h>

namespace oklt {
using namespace oklt;
using namespace clang;

namespace {
constexpr int MAX_N_PARAMS = 4;

tl::expected<int, Error> parseInt(const std::string& str) {
    char* p;
    auto tileSize = strtol(str.c_str(), &p, 10);
    if (!p) {
        return tl::make_unexpected(Error{std::error_code(), "Tile size is not an integer"});
    }
    return tileSize;
}

// TODO: parse dimension
tl::expected<LoopType, Error> parseLoopType(const std::string& str) {
    Error err{std::error_code(), "Tile loop type parse error"};
    if (str == "@outer") {
        return LoopType::Outer;
    } else if (str == "@inner") {
        return LoopType::Inner;
    }
    return tl::make_unexpected(err);
}

tl::expected<Loop, Error> parseLoop(const std::string& str) {
    Error err{std::error_code(), "Tile loop parse error"};

    auto nonDimLoopType = parseLoopType(str);
    // @inner or @outer (without dimensions)
    if (nonDimLoopType) {
        return Loop{nonDimLoopType.value(), Dim::X};  // x - default dimension
    }

    auto lpar_pos = str.find('(');
    auto rpar_pos = str.find(')');
    if (lpar_pos == std::string::npos || rpar_pos == std::string::npos || (lpar_pos > rpar_pos)) {
        return tl::make_unexpected(err);
    }
    auto idxNStr = str.substr(lpar_pos + 1, rpar_pos - lpar_pos);
    auto loopTypeStr = str.substr(lpar_pos);
    llvm::outs() << "[DEBUG] idxNStr: " << idxNStr << ", loopTypeStr: " << loopTypeStr << "\n";
    auto dimIdx = parseInt(idxNStr);
    auto loopType = parseLoopType(loopTypeStr);
    if (!dimIdx || !loopType) {
        return tl::make_unexpected(err);
    }
    if (dimIdx.value() < 0 || dimIdx.value() > 2) {
        return tl::make_unexpected(Error{std::error_code(), "Loop argument must be 0, 1, 2"});
    }
    return Loop{loopType.value(), static_cast<Dim>(dimIdx.value())};
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

}  // namespace

bool parseTileAttribute(const clang::Attr* a, SessionStage& s) {
    auto tileParamsStr = parseOKLAttributeParamsStr(a);
    if (!tileParamsStr.has_value()) {
        s.pushError(tileParamsStr.error());
        return false;
    }

    auto nParams = tileParamsStr->size();
    if (nParams > MAX_N_PARAMS || nParams < 1) {
        s.pushError(std::error_code(), "Tile has 1 to 4 parameters");
        return false;
    }

    // Parse all parameters:
    auto tileSize = parseInt(tileParamsStr.value()[0]);  // tileParamsStr is not empty for sure
    if (!tileSize.has_value()) {
        s.pushError(tileSize.error());
        return false;
    }

    std::vector<bool> checksStack;
    std::vector<Loop> loopsStack;

    // TODO: rewrite this ugly parsing if possible
    for (int i = 1; i < nParams; ++i) {
        auto currentParamStr = tileParamsStr.value()[i];
        // Parse check
        if (auto check = parseCheck(currentParamStr)) {
            checksStack.push_back(check.value());
            continue;
        }

        // Parse loop
        if (auto loopType = parseLoop(currentParamStr)) {
            loopsStack.push_back(loopType.value());
            continue;
        }

        s.pushError(std::error_code(), "Can't parse tile parameter: " + currentParamStr);
        return false;
    }

    // Verify number of parameters
    if (checksStack.size() > 1) {
        s.pushError(std::error_code(), "More than one tile check parameters");
        return false;
    }

    if (loopsStack.size() > 2) {
        s.pushError(std::error_code(), "More than two tile loop identifiers");
        return false;
    }

    TileParams tileParams{
        .tileSize = tileSize.value(),
        .firstLoop = loopsStack.size() > 0 ? loopsStack[0] : Loop{},
        .secondLoop = loopsStack.size() > 1 ? loopsStack[1] : Loop{},
        .check = checksStack.size() > 0 ? checksStack.front() : true,
    };

    // Outer can't be after inner:
    if (tileParams.firstLoop.type == LoopType::Inner &&
        tileParams.secondLoop.type == LoopType::Outer) {
        s.pushError(std::error_code(), "Cannot have [@inner] loop outside of an [@outer] loop");
        return false;
    }

    auto ctxKey = util::pointerToStr(static_cast<const void*>(a));
    s.tryEmplaceUserCtx<TileParams>(ctxKey, tileParams);

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] parsed tile parameters: " << ctxKey
                 << ": {tile size: " << tileParams.tileSize
                 << ", first loop: " << static_cast<int>(tileParams.firstLoop.type)
                 << " with dim: " << static_cast<int>(tileParams.firstLoop.dim)
                 << ", second loop: " << static_cast<int>(tileParams.secondLoop.type)
                 << " with dim: " << static_cast<int>(tileParams.secondLoop.dim)
                 << ", check: " << tileParams.check << "}\n";
#endif
    return true;
}
}  // namespace oklt