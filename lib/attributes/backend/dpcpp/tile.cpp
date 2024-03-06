#include "attributes/frontend/params/tile.h"
#include <oklt/core/kernel_metadata.h>
#include <oklt/util/string_utils.h>
#include "attributes/attribute_names.h"
#include "attributes/backend/dpcpp/common.h"
#include "attributes/frontend/params/tile.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

namespace {
using namespace oklt;

std::string getTiledVariableName(const OklLoopInfo& forLoop) {
    auto& meta = forLoop.metadata;
    return "_occa_tiled_" + meta.var.name;
}

std::string buildIinnerOuterLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = dpcpp::getIdxVariable(loop);
    auto& meta = forLoop.metadata;
    auto op = meta.IsInc() ? "+" : "-";

    std::string res;
    if (meta.isUnary()) {
        res = std::move(util::fmt("{} {} = ({}) {} (({}) * {});",
                                  meta.var.type,
                                  tiledVar,
                                  meta.range.start,
                                  op,
                                  params->tileSize,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = ({}) {} ((({}) * {}) * {});",
                                  meta.var.type,
                                  tiledVar,
                                  meta.range.start,
                                  op,
                                  params->tileSize,
                                  meta.inc.val,
                                  idx)
                            .value());
    }
    ++openedScopeCounter;
    return "{" + res;
}

std::string buildInnerOuterLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter) {
    static_cast<void>(params);
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = dpcpp::getIdxVariable(loop);
    auto& meta = forLoop.metadata;
    auto op = meta.IsInc() ? "+" : "-";

    std::string res;
    if (meta.isUnary()) {
        res = std::move(
            util::fmt("{} {} = {} {} {};", meta.var.type, meta.var.name, tiledVar, op, idx)
                .value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});",
                                  meta.var.type,
                                  meta.var.name,
                                  tiledVar,
                                  op,
                                  meta.inc.val,
                                  idx)
                            .value());
    }
    ++openedScopeCounter;
    return "{" + res;  // Open new scope
}

std::string buildRegularLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto& meta = forLoop.metadata;
    auto assignUpdate = meta.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(meta.condition.op);

    auto res = util::fmt("for({} {} = {}; {} {} {}; {} {} ({}))",
                         meta.var.type,
                         tiledVar,
                         meta.range.start,
                         tiledVar,
                         cmpOpStr,
                         meta.range.end,
                         tiledVar,
                         assignUpdate,
                         params->tileSize)
                   .value();  // shouldn't fail

    ++openedScopeCounter;
    return res + " {";  // Open new scope (Note: after line unlike @outer and @inner)
}

std::string buildRegularLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto& meta = forLoop.metadata;
    auto op = meta.IsInc() ? "+" : "-";
    auto cmp = meta.IsInc() ? "<" : ">";

    std::string res;
    if (meta.isUnary()) {
        auto unaryStr = getUnaryStr(meta.inc.op.uo, meta.var.name);  // ++i/i++/--i/i--
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {})",
                        meta.var.type,
                        meta.var.name,
                        tiledVar,
                        meta.var.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        unaryStr)
                  .value();
    } else {
        auto assignUpdate = meta.IsInc() ? "+=" : "-=";
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {} {} {})",
                        meta.var.type,
                        meta.var.name,
                        tiledVar,
                        meta.var.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        meta.var.name,
                        assignUpdate,
                        meta.inc.val)
                  .value();
    }
    return res;
}

std::string buildLoopIdxLine(const OklLoopInfo& forLoop,
                             const TileParams* params,
                             const LoopOrder& ord,
                             int& openedScopeCounter) {
    // TODO: this logic should be based on first or second loop, not inner/outer/regular
    static std::map<std::tuple<AttributedLoopType, LoopOrder>,
                    std::function<std::string(
                        const OklLoopInfo&, const AttributedLoop&, const TileParams*, int&)>>
        mapping{
            {{AttributedLoopType::Inner, LoopOrder::First}, buildIinnerOuterLoopIdxLineFirst},
            {{AttributedLoopType::Outer, LoopOrder::First}, buildIinnerOuterLoopIdxLineFirst},
            {{AttributedLoopType::Regular, LoopOrder::First}, buildRegularLoopIdxLineFirst},
            {{AttributedLoopType::Inner, LoopOrder::Second}, buildInnerOuterLoopIdxLineSecond},
            {{AttributedLoopType::Outer, LoopOrder::Second}, buildInnerOuterLoopIdxLineSecond},
            {{AttributedLoopType::Regular, LoopOrder::Second}, buildRegularLoopIdxLineSecond},
        };
    auto& loop = ord == LoopOrder::First ? params->firstLoop : params->secondLoop;
    return mapping[{loop.type, ord}](forLoop, loop, params, openedScopeCounter);
}

std::string buildCheckLine(const OklLoopInfo& forLoop,
                           const TileParams* tileParams,
                           int& openedScopeCounter) {
    if (!tileParams->check) {
        return "";
    }
    auto& meta = forLoop.metadata;
    auto cmpStr = getCondCompStr(meta.condition.op);

    // TODO: parse cmp operator
    auto res = util::fmt("if ({} {} {})", meta.var.name, cmpStr, meta.range.end).value();
    return res;
}

// TODO: add check handling
std::string buildPreffixTiledCode(const OklLoopInfo& forLoopMetaData,
                                  const TileParams* tileParams,
                                  int& openedScopeCounter) {
    std::string res;
    res += buildLoopIdxLine(forLoopMetaData, tileParams, LoopOrder::First, openedScopeCounter);
    res += buildLoopIdxLine(forLoopMetaData, tileParams, LoopOrder::Second, openedScopeCounter);
    res += buildCheckLine(forLoopMetaData, tileParams, openedScopeCounter);
    return res;
}

HandleResult handleTileAttribute(const clang::Attr& a,
                                 const clang::ForStmt& forStmt,
                                 const TileParams* params,
                                 SessionStage& s) {
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@tile params nullptr"});
    }

    auto& astCtx = s.getCompiler().getASTContext();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@tile: failed to fetch loop meta data from sema"});
    }
#ifdef TRANSPILER_DEBUG_LOG
    const auto& md = loopInfo->metadata;
    llvm::outs() << "[DEBUG] Handle @tile. Parsed for loop: Init(" << "type: " << toString(md.type)
                 << ", name: " << md.var.name << ", initValue: " << md.range.start
                 << "), Cond(rhsExpr: " << md.range.end << "), Inc(rhsInc: " << md.inc.val
                 << ", isUnary: " << md.isUnary() << ")\n";
#endif

    int openedScopeCounter = 0;
    auto prefixCode = buildPreffixTiledCode(*loopInfo, params, openedScopeCounter);
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    return replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);
}

__attribute__((constructor)) void registerDpcppTileAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, TILE_ATTR_NAME}, makeSpecificAttrHandle(handleTileAttribute));

    if (!ok) {
        llvm::errs() << "failed to register" << TILE_ATTR_NAME
                     << "attribute handler for DPCPP backend\n";
    }
}
}  // namespace
