#include "attributes/frontend/params/tile.h"
#include <clang/AST/Decl.h>
#include <oklt/util/string_utils.h>
#include <functional>
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "handle.h"
#include "attributes/utils/code_gen.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include <oklt/core/kernel_metadata.h>

namespace oklt::cuda_subset {
using namespace clang;
namespace {

std::string buildLoopIdxLine(const LoopMetaData& forLoop,
                             const TileParams* params,
                             const LoopOrder& ord,
                             int& openedScopeCounter) {
    // TODO: this logic should be based on first or second loop, not inner/outer/regular
    static std::map<std::tuple<LoopType, LoopOrder>,
                    std::function<std::string(
                        const LoopMetaData&, const AttributedLoop&, const TileParams*, int&)>>
        mapping{
            {{LoopType::Inner, LoopOrder::First}, tile::buildIinnerOuterLoopIdxLineFirst},
            {{LoopType::Outer, LoopOrder::First}, tile::buildIinnerOuterLoopIdxLineFirst},
            {{LoopType::Regular, LoopOrder::First}, tile::buildRegularLoopIdxLineFirst},
            {{LoopType::Inner, LoopOrder::Second}, tile::buildInnerOuterLoopIdxLineSecond},
            {{LoopType::Outer, LoopOrder::Second}, tile::buildInnerOuterLoopIdxLineSecond},
            {{LoopType::Regular, LoopOrder::Second}, tile::buildRegularLoopIdxLineSecond},
        };
    auto& loop = ord == LoopOrder::First ? params->firstLoop : params->secondLoop;
    return mapping[{loop.type, ord}](forLoop, loop, params, openedScopeCounter);
}

std::string buildCheckLine(const LoopMetaData& forLoop,
                           const TileParams* tileParams,
                           int& openedScopeCounter) {
    if (!tileParams->check) {
        return "";
    }
    auto cmpStr = getCondCompStr(forLoop.condition.op);

    // TODO: parse cmp operator
    auto res = util::fmt("if ({} {} {})", forLoop.name, cmpStr, forLoop.range.end).value();
    return res;
}

// TODO: add check handling
std::string buildPreffixTiledCode(const LoopMetaData& forLoopMetaData,
                                  const TileParams* tileParams,
                                  int& openedScopeCounter) {
    std::string res;
    res += buildLoopIdxLine(forLoopMetaData, tileParams, LoopOrder::First, openedScopeCounter);
    res += buildLoopIdxLine(forLoopMetaData, tileParams, LoopOrder::Second, openedScopeCounter);
    res += buildCheckLine(forLoopMetaData, tileParams, openedScopeCounter);
    return res;
}

}  // namespace

bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s) {
    auto usrCtxKey = util::pointerToStr(a);
    auto tileParams = std::any_cast<TileParams>(s.getUserCtx(usrCtxKey));
    if (tileParams == nullptr) {
        s.pushError(std::error_code(), "No @tile params in user context");
        return false;
    }

    auto& astCtx = s.getCompiler().getASTContext();

    if (!isa<ForStmt>(d)) {
        s.pushError(std::error_code(), "@tile can be applied only to for loop");
        return false;
    }
    const auto* forStmt = dyn_cast<ForStmt>(d);
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto forLoopMetaData = sema.getLoopMetaData(forStmt);
    if (!forLoopMetaData) {
        s.pushError(std::error_code(), "@tile: failed to fetch loop meta data from sema");
        return false;
    }
    // auto forLoopMetaData = ParseForStmt(const_cast<ForStmt*>(forStmt), astCtx);

    int openedScopeCounter = 0;
    auto prefixCode = buildPreffixTiledCode(forLoopMetaData.value(), tileParams, openedScopeCounter);
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @tile. Parsed for loop: Init("
                 << "type: " << forLoopMetaData->type << ", name: " << forLoopMetaData->name
                 << ", initValue: " << forLoopMetaData->range.start
                 << "), Cond(rhsExpr: " << forLoopMetaData->range.end
                 << "), Inc(rhsInc: " << forLoopMetaData->inc.val
                 << ", isUnary: " << forLoopMetaData->isUnary() << ")\n";
#endif
    return true;
}
}  // namespace oklt::cuda_subset
