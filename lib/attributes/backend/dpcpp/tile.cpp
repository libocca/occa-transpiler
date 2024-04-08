#include <oklt/core/kernel_metadata.h>
#include <oklt/util/string_utils.h>

#include "attributes/attribute_names.h"
#include "attributes/backend/dpcpp/common.h"
#include "attributes/frontend/params/tile.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/kernel_utils.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/range_to_string.h"

#include <clang/Rewrite/Core/Rewriter.h>

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

std::string getTiledVariableName(const OklLoopInfo& forLoop) {
    return "_occa_tiled_" + forLoop.var.name;
}

std::string buildIinnerOuterLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter,
                                             oklt::Rewriter& rewriter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = dpcpp::getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = ({}) {} (({}) * {});\n",
                                  forLoop.var.typeName,
                                  tiledVar,
                                  getLatestSourceText(forLoop.range.start, rewriter),
                                  op,
                                  params->tileSize,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = ({}) {} ((({}) * {}) * {});\n",
                                  forLoop.var.typeName,
                                  tiledVar,
                                  getLatestSourceText(forLoop.range.start, rewriter),
                                  op,
                                  params->tileSize,
                                  getLatestSourceText(forLoop.inc.val, rewriter),
                                  idx)
                            .value());
    }

    ++openedScopeCounter;
    return " {\n" + res;
}

std::string buildInnerOuterLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter,
                                             oklt::Rewriter& rewriter) {
    static_cast<void>(params);
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = dpcpp::getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(
            util::fmt(
                "{} {} = {} {} {};", forLoop.var.typeName, forLoop.var.name, tiledVar, op, idx)
                .value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});\n",
                                  forLoop.var.typeName,
                                  forLoop.var.name,
                                  tiledVar,
                                  op,
                                  getLatestSourceText(forLoop.inc.val, rewriter),
                                  idx)
                            .value());
    }

    ++openedScopeCounter;
    return " {\n" + res;  // Open new scope
}

std::string buildRegularLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter,
                                         oklt::Rewriter& rewriter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(forLoop.condition.op);

    auto res = util::fmt("for ({} {} = {}; {} {} {}; {} {} ({}))",
                         forLoop.var.typeName,
                         tiledVar,
                         getLatestSourceText(forLoop.range.start, rewriter),
                         tiledVar,
                         cmpOpStr,
                         getLatestSourceText(forLoop.range.end, rewriter),
                         tiledVar,
                         assignUpdate,
                         params->tileSize)
                   .value();  // shouldn't fail

    // Open new scope (Note: after line unlike @outer and @inner)
    ++openedScopeCounter;
    return res + " {\n";
}

std::string buildRegularLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter,
                                          oklt::Rewriter& rewriter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto op = forLoop.IsInc() ? "+" : "-";
    auto cmp = forLoop.IsInc() ? "<" : ">";

    std::string res;
    if (forLoop.isUnary()) {
        auto unaryStr = getUnaryStr(forLoop.inc.op.uo, forLoop.var.name);  // ++i/i++/--i/i--
        res = util::fmt("for ({} {} = {}; {} {} ({} {} ({})); {})",
                        forLoop.var.typeName,
                        forLoop.var.name,
                        tiledVar,
                        forLoop.var.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        unaryStr)
                  .value();
    } else {
        auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
        res = util::fmt("for ({} {} = {}; {} {} ({} {} ({})); {} {} {})",
                        forLoop.var.typeName,
                        forLoop.var.name,
                        tiledVar,
                        forLoop.var.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        forLoop.var.name,
                        assignUpdate,
                        getLatestSourceText(forLoop.inc.val, rewriter))
                  .value();
    }

    auto& stmt = forLoop.stmt;
    if (params->check || !llvm::isa<clang::CompoundStmt>(stmt.getBody())) {
        ++openedScopeCounter;
        res += " {\n";
    }

    return res;
}

std::string buildLoopIdxLine(const OklLoopInfo& forLoop,
                             const TileParams* params,
                             const LoopOrder& ord,
                             int& openedScopeCounter,
                             oklt::Rewriter& rewriter) {
    static std::map<
        std::tuple<LoopType, LoopOrder>,
        std::function<std::string(
            const OklLoopInfo&, const AttributedLoop&, const TileParams*, int&, oklt::Rewriter&)>>
        mapping{
            {{LoopType::Inner, LoopOrder::First}, buildIinnerOuterLoopIdxLineFirst},
            {{LoopType::Outer, LoopOrder::First}, buildIinnerOuterLoopIdxLineFirst},
            {{LoopType::Regular, LoopOrder::First}, buildRegularLoopIdxLineFirst},
            {{LoopType::Inner, LoopOrder::Second}, buildInnerOuterLoopIdxLineSecond},
            {{LoopType::Outer, LoopOrder::Second}, buildInnerOuterLoopIdxLineSecond},
            {{LoopType::Regular, LoopOrder::Second}, buildRegularLoopIdxLineSecond},
        };
    auto& loop = ord == LoopOrder::First ? params->firstLoop : params->secondLoop;
    return mapping[{loop.type, ord}](forLoop, loop, params, openedScopeCounter, rewriter);
}

std::string buildCheckLine(const OklLoopInfo& forLoop,
                           const TileParams* tileParams,
                           int& openedScopeCounter,
                           oklt::Rewriter& rewriter) {
    if (!tileParams->check) {
        return "";
    }

    auto cmpStr = getCondCompStr(forLoop.condition.op);

    auto res = util::fmt("if ({} {} {})",
                         forLoop.var.name,
                         cmpStr,
                         getLatestSourceText(forLoop.range.end, rewriter))
                   .value();

    auto& stmt = forLoop.stmt;
    if (!llvm::isa<clang::CompoundStmt>(stmt.getBody())) {
        ++openedScopeCounter;
        res += " {\n";
    }

    return res;
}

std::string buildPreffixTiledCode(const OklLoopInfo& forLoop,
                                  const TileParams* tileParams,
                                  int& openedScopeCounter,
                                  oklt::Rewriter& rewriter) {
    std::string res;
    res += buildLoopIdxLine(forLoop, tileParams, LoopOrder::First, openedScopeCounter, rewriter);
    res += buildLoopIdxLine(forLoop, tileParams, LoopOrder::Second, openedScopeCounter, rewriter);
    res += buildCheckLine(forLoop, tileParams, openedScopeCounter, rewriter);
    return res;
}

HandleResult handleTileAttribute(const clang::Attr& a,
                                 const clang::ForStmt& forStmt,
                                 const TileParams* params,
                                 SessionStage& s) {
    SPDLOG_DEBUG("Handle [@tile] attribute");

    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@tile params nullptr"});
    }

    auto& astCtx = s.getCompiler().getASTContext();
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
        afterRBraceCode += dpcpp::SYNC_THREADS_BARRIER + ";";
    }

    handleChildAttr(forStmt, NOBARRIER_ATTR_NAME, s);

    return replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, afterRBraceCode, s);
}

__attribute__((constructor)) void registerDpcppTileAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, TILE_ATTR_NAME}, makeSpecificAttrHandle(handleTileAttribute));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", TILE_ATTR_NAME);
    }
}
}  // namespace
