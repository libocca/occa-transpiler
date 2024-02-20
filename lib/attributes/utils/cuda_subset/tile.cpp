#include "attributes/frontend/params/tile.h"
#include <clang/AST/Decl.h>
#include <oklt/util/string_utils.h>
#include <functional>
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "attributes/utils/loop_meta_data.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "handle.h"

namespace oklt::cuda_subset {
using namespace clang;
namespace {

std::string getLoopIdxLine(const LoopMetadata& forLoop,
                           const TileParams* params,
                           const LoopOrder& ord,
                           int& openedScopeCounter) {
    // TODO: this logic should be based on first or second loop, not inner/outer/regular
    static std::map<std::tuple<LoopType, LoopOrder>,
                    std::function<std::string(
                        const LoopMetadata&, const AttributedLoop&, const TileParams*, int&)>>
        mapping{
            {{LoopType::Inner, LoopOrder::First}, tile::innerOuterLoopIdxLineFirst},
            {{LoopType::Outer, LoopOrder::First}, tile::innerOuterLoopIdxLineFirst},
            {{LoopType::Regular, LoopOrder::First}, tile::regularLoopIdxLineFirst},
            {{LoopType::Inner, LoopOrder::Second}, tile::innerOuterLoopIdxLineSecond},
            {{LoopType::Outer, LoopOrder::Second}, tile::innerOuterLoopIdxLineSecond},
            {{LoopType::Regular, LoopOrder::Second}, tile::regularLoopIdxLineSecond},
        };
    auto& loop = ord == LoopOrder::First ? params->firstLoop : params->secondLoop;
    return mapping[{loop.type, ord}](forLoop, loop, params, openedScopeCounter);
}

std::string getCheckLine(const LoopMetadata& forLoop,
                         const TileParams* tileParams,
                         int& openedScopeCounter) {
    if (!tileParams->check) {
        return "";
    }
    auto cmpStr = getCondCompStr(forLoop);

    // TODO: parse cmp operator
    auto res = util::fmt("if ({} {} {})", forLoop.name, cmpStr, forLoop.range.end).value();
    return res;
}

// TODO: add check handling
std::string buildPreffixTiledCode(const LoopMetadata& forLoopMetaData,
                                  const TileParams* tileParams,
                                  int& openedScopeCounter) {
    std::string res;
    res += getLoopIdxLine(forLoopMetaData, tileParams, LoopOrder::First, openedScopeCounter);
    res += getLoopIdxLine(forLoopMetaData, tileParams, LoopOrder::Second, openedScopeCounter);
    res += getCheckLine(forLoopMetaData, tileParams, openedScopeCounter);
    return res;
}

std::string buildSuffixTiledCode(int& openedScopeCounter) {
    std::string res;
    // Close all opened scopes
    while (openedScopeCounter--) {
        res += "}";
    }
    return res;
}

}  // namespace

bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s) {
    auto usrCtxKey = util::pointerToStr(static_cast<const void*>(a));
    auto tileParams = std::any_cast<TileParams>(s.getUserCtx(usrCtxKey));
    if (tileParams == nullptr) {
        s.pushError(std::error_code(), "No tile params in user context");
        return false;
    }

    auto& astCtx = s.getCompiler().getASTContext();

    if (!isa<ForStmt>(d)) {
        s.pushError(std::error_code(), "Tile can be applied to only for loop");
        return false;
    }
    const auto* forStmt = dyn_cast<ForStmt>(d);
    auto forLoopMetaData = ParseForStmt(const_cast<ForStmt*>(forStmt), astCtx);

    int openedScopeCounter = 0;
    auto prefixCode = buildPreffixTiledCode(forLoopMetaData, tileParams, openedScopeCounter);
    auto suffixCode = buildSuffixTiledCode(openedScopeCounter);

    auto& rewriter = s.getRewriter();

    // Remove attribute + for loop
    SourceRange range;
    range.setBegin(a->getRange().getBegin().getLocWithOffset(-2));  // TODO: remove magic number
    range.setEnd(forStmt->getRParenLoc());
    rewriter.RemoveText(range);

    // Insert preffix
    rewriter.InsertText(forStmt->getRParenLoc(), prefixCode);

    // Insert suffix
    rewriter.InsertText(forStmt->getEndLoc(),
                        suffixCode);  // TODO: seems to not work correclty for for loop without {}

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle Tile. Parsed for loop: Init("
                 << "type: " << forLoopMetaData.type << ", name: " << forLoopMetaData.name
                 << ", initValue: " << forLoopMetaData.range.start
                 << "), Cond(rhsExpr: " << forLoopMetaData.range.end
                 << "), Inc(rhsInc: " << forLoopMetaData.inc.val
                 << ", isUnary: " << forLoopMetaData.isUnary() << ")\n";
#endif
    return true;
}
}  // namespace oklt::cuda_subset
