#include <clang/AST/Decl.h>
#include <oklt/attributes/common/attribute_utils.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/attributes/common/parsers/tile.hpp>

namespace oklt {
using namespace oklt;
using namespace clang;

namespace {
constexpr int MAX_N_PARAMS = 4;

tl::expected<int, Error> parseTileSize(const std::string& str) {
    char* p;
    auto tileSize = strtol(str.c_str(), &p, 10);
    if (!p) {
        return tl::make_unexpected(Error{std::error_code(), "Tile size is not an integer"});
    }
    return tileSize;
}

tl::expected<LoopType, Error> parseLoopType(const std::string& str) {
    if (str == "@outer") {
        return LoopType::Outer;
    } else if (str == "@inner") {
        return LoopType::Inner;
    }
    return tl::make_unexpected(Error{std::error_code(), "Tile loop type parse error"});
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
    auto tileSize = parseTileSize(tileParamsStr.value()[0]);  // tileParamsStr is not empty for sure
    if (!tileSize.has_value()) {
        s.pushError(tileSize.error());
    }

    // Default values
    tl::expected<LoopType, Error> firstLoopType = LoopType::Regular;
    tl::expected<LoopType, Error> secondLoopType = LoopType::Regular;
    tl::expected<bool, Error> check = false;
    bool firstLoopIsSet = false;
    bool secondLoopIsSet = false;
    bool checkIsSet = false;

    Error err;
    // TODO: rewrite this ugly parsing if possible
    for (int i = 1; i < nParams; ++i) {
        // Parse check
        auto currentCheck = parseCheck(tileParamsStr.value()[i]);
        if (currentCheck) {
            if (checkIsSet) {
                err = Error{std::error_code(), "check is set two times"};
                break;
            } else {
                check = std::move(currentCheck);
                checkIsSet = true;
                continue;
            }
        } else {
            if (firstLoopIsSet && secondLoopIsSet) {
                err = check.error();
                break;
            }
        }

        auto loopType = parseLoopType(tileParamsStr.value()[i]);
        if (loopType) {
            if (!firstLoopIsSet) {
                firstLoopType = std::move(loopType);
                firstLoopIsSet = true;
            } else if (!secondLoopIsSet) {
                secondLoopType = std::move(loopType);
                secondLoopIsSet = true;
            } else {
                err = Error{std::error_code(), "maximum two @inner and/or @outer allowed"};
            }
        } else {
            err = loopType.error();
            break;
        }
    }

    if (!err.desc.empty()) {
        s.pushError(err);
        return false;
    }
    TileParams tileParams{
        tileSize.value(), firstLoopType.value(), secondLoopType.value(), check.value()};
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] parsed tile parameters: {tile size: " << tileParams.tileSize
                 << ", first loop: " << static_cast<int>(tileParams.firstLoopType)
                 << ", second loop: " << static_cast<int>(tileParams.secondLoopType)
                 << ", check: " << tileParams.check << "}\n";
#endif
    return true;
}
}  // namespace oklt