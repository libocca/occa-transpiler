#include <oklt/core/kernel_metadata.h>
#include <oklt/util/string_utils.h>

#include "attributes/frontend/params/loop.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "attributes/utils/tile_utils.h"

#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "tl/expected.hpp"

#include <clang/AST/Decl.h>

#include <functional>

#include <clang/AST/Decl.h>

#include <functional>

namespace oklt::cuda_subset {
using namespace clang;
namespace {

std::string buildLoopIdxLine(const OklLoopInfo& forLoop,
                             const TileParams* params,
                             const LoopOrder& ord,
                             int& openedScopeCounter) {
    // TODO: this logic should be based on first or second loop, not inner/outer/regular
    static std::map<std::tuple<AttributedLoopType, LoopOrder>,
                    std::function<std::string(
                        const OklLoopInfo&, const AttributedLoop&, const TileParams*, int&)>>
        mapping{
            {{AttributedLoopType::Inner, LoopOrder::First}, tile::buildIinnerOuterLoopIdxLineFirst},
            {{AttributedLoopType::Outer, LoopOrder::First}, tile::buildIinnerOuterLoopIdxLineFirst},
            {{AttributedLoopType::Regular, LoopOrder::First}, tile::buildRegularLoopIdxLineFirst},
            {{AttributedLoopType::Inner, LoopOrder::Second},
             tile::buildInnerOuterLoopIdxLineSecond},
            {{AttributedLoopType::Outer, LoopOrder::Second},
             tile::buildInnerOuterLoopIdxLineSecond},
            {{AttributedLoopType::Regular, LoopOrder::Second}, tile::buildRegularLoopIdxLineSecond},
        };
    auto& loop = ord == LoopOrder::First ? params->firstLoop : params->secondLoop;
    return mapping[{loop.type, ord}](forLoop, loop, params, openedScopeCounter);
}

std::string buildCheckLine(const OklLoopInfo& forLoop,
                           const TileParams* params,
                           int& openedScopeCounter) {
    if (!params->check) {
        return "";
    }
    auto& meta = forLoop.metadata;
    auto cmpStr = getCondCompStr(meta.condition.op);

    // TODO: parse cmp operator
    auto res = util::fmt("if ({} {} {})", meta.var.name, cmpStr, meta.range.end).value();
    return res;
}

// TODO: add check handling
std::string buildPreffixTiledCode(const OklLoopInfo& forLoop,
                                  const TileParams* params,
                                  int& openedScopeCounter) {
    std::string res;
    res += buildLoopIdxLine(forLoop, params, LoopOrder::First, openedScopeCounter);
    res += buildLoopIdxLine(forLoop, params, LoopOrder::Second, openedScopeCounter);
    res += buildCheckLine(forLoop, params, openedScopeCounter);
    return res;
}

}  // namespace

HandleResult handleTileAttribute(const clang::Attr& a,
                                 const clang::ForStmt& forStmt,
                                 const TileParams* params,
                                 SessionStage& s) {
    auto& astCtx = s.getCompiler().getASTContext();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@tile: failed to fetch loop meta data from sema"});
    }

    auto updatedParams = tileParamsHandleAutoDims(*params, *loopInfo);
    if (!updatedParams) {
        return tl::make_unexpected(updatedParams.error());
    }
    int openedScopeCounter = 0;
    auto prefixCode = buildPreffixTiledCode(*loopInfo, &updatedParams.value(), openedScopeCounter);
    auto suffixCode = buildCloseScopes(openedScopeCounter);

#ifdef TRANSPILER_DEBUG_LOG
    const auto& md = loopInfo->metadata;
    llvm::outs() << "[DEBUG] Handle @tile. Parsed for loop: Init("
                 << ", name: " << md.var.name << ", initValue: " << md.range.start
                 << "), Cond(rhsExpr: " << md.range.end << "), Inc(rhsInc: " << md.inc.val
                 << ", isUnary: " << md.isUnary() << ")\n";
#endif
    return replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);
}
}  // namespace oklt::cuda_subset
