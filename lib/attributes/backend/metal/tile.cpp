#include "attributes/frontend/params/tile.h"
#include "attributes/backend/metal/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

// Produces something like: int _occa_tiled_i = init +- ((tileSize * inc) * _occa_group_position.x);
//                      or: int _occa_tiled_i = init +- ((tileSize * inc) *
//                      _occa_thread_position.x);
std::string buildIinnerOuterLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter,
                                             oklt::Rewriter& rewriter) {
    auto tiledVar = metal::getTiledVariableName(forLoop);
    auto idx = metal::getIdxVariable(loop);
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

// Produces something like: int i = _occa_tiled_i +- (inc * _occa_group_position.x);
//                      or: int i = _occa_tiled_i +- (inc * _occa_thread_position.x);
std::string buildInnerOuterLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter,
                                             oklt::Rewriter& rewriter) {
    static_cast<void>(params);
    auto tiledVar = metal::getTiledVariableName(forLoop);
    auto idx = metal::getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(
            util::fmt(
                "{} {} = {} {} {};\n", forLoop.var.typeName, forLoop.var.name, tiledVar, op, idx)
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

// Produces something like:
//      for (int _occa_tiled_i = start; _occa_tiled_i < end; _occa_tiled_i += tileSize) {
// or:  for (int _occa_tiled_i = start; _occa_tiled_i > end; _occa_tiled_i -= tileSize) {
std::string buildRegularLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter,
                                         oklt::Rewriter& rewriter) {
    auto tiledVar = metal::getTiledVariableName(forLoop);
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

    ++openedScopeCounter;
    return res + " {\n";  // Open new scope (Note: after line unlike @outer and @inner)
}

// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); ++i)
// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); i+=inc)
//                      or: for (int i = _occa_tiled_i; i > (_occa_tiled_i - tileSize); --i)
//                      or: for (int i = _occa_tiled_i; i > (_occa_tiled_i - tileSize); i-=inc)
std::string buildRegularLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter,
                                          oklt::Rewriter& rewriter) {
    auto tiledVar = metal::getTiledVariableName(forLoop);
    auto& stmt = forLoop.stmt;
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
                           const TileParams* params,
                           int& openedScopeCounter,
                           oklt::Rewriter& rewriter) {
    if (!params->check) {
        return "";
    }
    auto cmpStr = getCondCompStr(forLoop.condition.op);

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

std::string buildPreffixTiledCode(const OklLoopInfo& forLoop,
                                  const TileParams* params,
                                  int& openedScopeCounter,
                                  oklt::Rewriter& rewriter) {
    std::string res;
    res += buildLoopIdxLine(forLoop, params, LoopOrder::First, openedScopeCounter, rewriter);
    res += buildLoopIdxLine(forLoop, params, LoopOrder::Second, openedScopeCounter, rewriter);
    res += buildCheckLine(forLoop, params, openedScopeCounter, rewriter);
    return res;
}

HandleResult handleTileAttribute(SessionStage& s,
                                 const clang::ForStmt& forStmt,
                                 const clang::Attr& a,
                                 const TileParams* params) {
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
        afterRBraceCode += metal::SYNC_THREADS_BARRIER + ";";
    }

    handleChildAttr(s, forStmt, NO_BARRIER_ATTR_NAME);

    return replaceAttributedLoop(s, forStmt, a, suffixCode, afterRBraceCode, prefixCode, false);
}

__attribute__((constructor)) void registerHIPTileAttrBackend() {
    auto ok = registerBackendHandler(TargetBackend::METAL, TILE_ATTR_NAME, handleTileAttribute);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register {} attribute handler", TILE_ATTR_NAME);
    }
}
}  // namespace
