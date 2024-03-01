#include "attributes/frontend/params/tile.h"
#include <oklt/core/kernel_metadata.h>
#include <oklt/util/string_utils.h>
#include "attributes/attribute_names.h"
#include "attributes/backend/dpcpp/common.h"
#include "attributes/frontend/params/tile.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

namespace {
using namespace oklt;

std::string getTiledVariableName(const LoopMetaData& forLoop) {
    return "_occa_tiled_" + forLoop.name;
}

std::string buildIinnerOuterLoopIdxLineFirst(const LoopMetaData& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = dpcpp::getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = ({}) {} (({}) * {});",
                                  forLoop.type,
                                  tiledVar,
                                  forLoop.range.start,
                                  op,
                                  params->tileSize,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = ({}) {} ((({}) * {}) * {});",
                                  forLoop.type,
                                  tiledVar,
                                  forLoop.range.start,
                                  op,
                                  params->tileSize,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    return res;
}

std::string buildInnerOuterLoopIdxLineSecond(const LoopMetaData& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter) {
    static_cast<void>(params);
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = dpcpp::getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(
            util::fmt("{} {} = {} {} {};", forLoop.type, forLoop.name, tiledVar, op, idx).value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});",
                                  forLoop.type,
                                  forLoop.name,
                                  tiledVar,
                                  op,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    ++openedScopeCounter;
    return "{" + res;  // Open new scope
}

std::string buildRegularLoopIdxLineFirst(const LoopMetaData& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(forLoop.condition.op);

    auto res = util::fmt("for({} {} = {}; {} {} {}; {} {} ({}))",
                         forLoop.type,
                         tiledVar,
                         forLoop.range.start,
                         tiledVar,
                         cmpOpStr,
                         forLoop.range.end,
                         tiledVar,
                         assignUpdate,
                         params->tileSize)
                   .value();  // shouldn't fail

    ++openedScopeCounter;
    return res + " {";  // Open new scope (Note: after line unlike @outer and @inner)
}

std::string buildRegularLoopIdxLineSecond(const LoopMetaData& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto op = forLoop.IsInc() ? "+" : "-";
    auto cmp = forLoop.IsInc() ? "<" : ">";

    std::string res;
    if (forLoop.isUnary()) {
        auto unaryStr = getUnaryStr(forLoop.inc.op.uo, forLoop.name);  // ++i/i++/--i/i--
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {})",
                        forLoop.type,
                        forLoop.name,
                        tiledVar,
                        forLoop.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        unaryStr)
                  .value();
    } else {
        auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {} {} {})",
                        forLoop.type,
                        forLoop.name,
                        tiledVar,
                        forLoop.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        forLoop.name,
                        assignUpdate,
                        forLoop.inc.val)
                  .value();
    }
    return res;
}

std::string buildLoopIdxLine(const LoopMetaData& forLoop,
                             const TileParams* params,
                             const LoopOrder& ord,
                             int& openedScopeCounter) {
    // TODO: this logic should be based on first or second loop, not inner/outer/regular
    static std::map<std::tuple<LoopType, LoopOrder>,
                    std::function<std::string(
                        const LoopMetaData&, const AttributedLoop&, const TileParams*, int&)>>
        mapping{
            {{LoopType::Inner, LoopOrder::First}, buildIinnerOuterLoopIdxLineFirst},
            {{LoopType::Outer, LoopOrder::First}, buildIinnerOuterLoopIdxLineFirst},
            {{LoopType::Regular, LoopOrder::First}, buildRegularLoopIdxLineFirst},
            {{LoopType::Inner, LoopOrder::Second}, buildInnerOuterLoopIdxLineSecond},
            {{LoopType::Outer, LoopOrder::Second}, buildInnerOuterLoopIdxLineSecond},
            {{LoopType::Regular, LoopOrder::Second}, buildRegularLoopIdxLineSecond},
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

HandleResult handleTileAttribute(const clang::Attr& a,
                                 const clang::ForStmt& forStmt,
                                 const TileParams* params,
                                 SessionStage& s) {
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@tile params nullptr"});
    }

    auto& astCtx = s.getCompiler().getASTContext();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto forLoopMetaData = sema.getLoopMetaData(forStmt);
    if (!forLoopMetaData) {
        return tl::make_unexpected(Error{{}, "@tile: failed to fetch loop meta data from sema"});
    }
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @tile. Parsed for loop: Init("
                 << "type: " << forLoopMetaData->type << ", name: " << forLoopMetaData->name
                 << ", initValue: " << forLoopMetaData->range.start
                 << "), Cond(rhsExpr: " << forLoopMetaData->range.end
                 << "), Inc(rhsInc: " << forLoopMetaData->inc.val
                 << ", isUnary: " << forLoopMetaData->isUnary() << ")\n";
#endif

    int openedScopeCounter = 0;
    auto prefixCode = buildPreffixTiledCode(forLoopMetaData.value(), params, openedScopeCounter);
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
