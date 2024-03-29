#include <oklt/core/kernel_metadata.h>
#include <oklt/util/string_utils.h>

#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/common.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "attributes/utils/kernel_utils.h"

#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/range_to_string.h"
#include "tl/expected.hpp"

#include <clang/Rewrite/Core/Rewriter.h>

#include <functional>

#include <spdlog/spdlog.h>

namespace {
using namespace clang;
using namespace oklt;

std::string buildLoopIdxLine(const OklLoopInfo& forLoop,
                             const TileParams* params,
                             const LoopOrder& ord,
                             int& openedScopeCounter,
                             clang::Rewriter& rewriter) {
    using namespace oklt::cuda_subset;

    // TODO: this logic should be based on first or second loop, not inner/outer/regular
    static std::map<
        std::tuple<LoopType, LoopOrder>,
        std::function<std::string(
            const OklLoopInfo&, const AttributedLoop&, const TileParams*, int&, clang::Rewriter&)>>
        mapping{
            {{LoopType::Inner, LoopOrder::First}, tile::buildIinnerOuterLoopIdxLineFirst},
            {{LoopType::Outer, LoopOrder::First}, tile::buildIinnerOuterLoopIdxLineFirst},
            {{LoopType::Regular, LoopOrder::First}, tile::buildRegularLoopIdxLineFirst},
            {{LoopType::Inner, LoopOrder::Second}, tile::buildInnerOuterLoopIdxLineSecond},
            {{LoopType::Outer, LoopOrder::Second}, tile::buildInnerOuterLoopIdxLineSecond},
            {{LoopType::Regular, LoopOrder::Second}, tile::buildRegularLoopIdxLineSecond},
        };
    auto& loop = ord == LoopOrder::First ? params->firstLoop : params->secondLoop;
    return mapping[{loop.type, ord}](forLoop, loop, params, openedScopeCounter, rewriter);
}

std::string buildCheckLine(const OklLoopInfo& forLoop,
                           const TileParams* params,
                           int& openedScopeCounter,
                           clang::Rewriter& rewriter) {
    if (!params->check) {
        return "";
    }
    auto cmpStr = getCondCompStr(forLoop.condition.op);

    // TODO: parse cmp operator
    auto res = util::fmt("if ({} {} {})",
                         forLoop.var.name,
                         cmpStr,
                         getLatestSourceText(forLoop.range.end, rewriter))
                   .value();

    auto& stmt = forLoop.stmt;
    if (!isa<clang::CompoundStmt>(stmt.getBody())) {
        ++openedScopeCounter;
        res += " {\n";
    }

    return res;
}

// TODO: add check handling
std::string buildPreffixTiledCode(const OklLoopInfo& forLoop,
                                  const TileParams* params,
                                  int& openedScopeCounter,
                                  clang::Rewriter& rewriter) {
    std::string res;
    res += buildLoopIdxLine(forLoop, params, LoopOrder::First, openedScopeCounter, rewriter);
    res += buildLoopIdxLine(forLoop, params, LoopOrder::Second, openedScopeCounter, rewriter);
    res += buildCheckLine(forLoop, params, openedScopeCounter, rewriter);
    return res;
}

}  // namespace

namespace oklt::cuda_subset {
using namespace clang;

HandleResult handleTileAttribute(const Attr& a,
                                 const ForStmt& forStmt,
                                 const TileParams* params,
                                 SessionStage& s) {
    SPDLOG_DEBUG("Handle [@tile] attribute");

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@tile: failed to fetch loop meta data from sema"});
    }

    auto updatedParams = *params;
    // Auto Axis in loopInfo are replaced with specific. TODO: maybe somehow update params earlier?
    updatedParams.firstLoop.axis = loopInfo->axis[0];
    updatedParams.secondLoop.axis = loopInfo->axis[1];

    int openedScopeCounter = 0;
    auto prefixCode =
        buildPreffixTiledCode(*loopInfo, &updatedParams, openedScopeCounter, s.getRewriter());
    auto suffixCode = buildCloseScopes(openedScopeCounter);
    std::string afterRBraceCode = "";
    if (loopInfo->shouldSync()) {
        afterRBraceCode += cuda_subset::SYNC_THREADS_BARRIER + ";";
    }

    handleChildAttr(forStmt, NOBARRIER_ATTR_NAME, s);

    return replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, afterRBraceCode, s);
}

}  // namespace oklt::cuda_subset
