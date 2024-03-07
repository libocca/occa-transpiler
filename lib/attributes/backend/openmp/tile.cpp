#include "attributes/frontend/params/tile.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/code_gen.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <oklt/core/kernel_metadata.h>
#include <oklt/util/string_utils.h>

#include <clang/Rewrite/Core/Rewriter.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string prefixText = "#pragma omp parallel for\n";
const std::string exclusiveNullText = "_occa_exclusive_index = 0;\n";
const std::string exclusiveIncText = "++_occa_exclusive_index;\n";

std::string getTiledVariableName(const OklLoopInfo& forLoop) {
    auto& meta = forLoop.metadata;
    return "_occa_tiled_" + meta.var.name;
}

std::string getScopesCloseStr(size_t& parenCnt) {
    std::string ret;
    while (parenCnt--) {
        ret += "}\n";
    }
    return ret;
}

std::string buildFirstLoopString([[maybe_unused]] const ForStmt& stmt,
                                 const OklLoopInfo& loopInfo,
                                 [[maybe_unused]] const TileParams* params,
                                 size_t& parenCnt) {
    auto tiledVar = getTiledVariableName(loopInfo);
    auto& meta = loopInfo.metadata;
    auto assignUpdate = meta.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(meta.condition.op);
    auto incValStr = params->tileSize;
    if (!meta.inc.val.empty()) {
        incValStr = util::fmt("({} * {})", params->tileSize, meta.inc.val).value();
    }

    auto ret = util::fmt("for ({} {} = {}; {} {} {}; {} {} {})",
                         meta.var.type,
                         tiledVar,
                         meta.range.start,
                         tiledVar,
                         cmpOpStr,
                         meta.range.end,
                         tiledVar,
                         assignUpdate,
                         incValStr)
                   .value();

    ++parenCnt;
    ret += " {\n";

    return ret;
}

std::string buildSecondLoopString([[maybe_unused]] const ForStmt& stmt,
                                  const OklLoopInfo& loopInfo,
                                  [[maybe_unused]] const TileParams* params,
                                  size_t& parenCnt) {
    auto tiledVar = getTiledVariableName(loopInfo);
    auto& meta = loopInfo.metadata;
    auto op = meta.IsInc() ? "+" : "-";
    auto cmp = meta.IsInc() ? "<" : ">";

    std::string ret;
    if (meta.isUnary()) {
        auto unaryStr = getUnaryStr(meta.inc.op.uo, meta.var.name);
        ret = util::fmt("for ({} {} = {}; {} {} ({} {} {}); {})",
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
        ret = util::fmt("for ({} {} = {}; {} {} ({} {} {}); {} {} {})",
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

    if (params->check || !isa<CompoundStmt>(stmt.getBody())) {
        ++parenCnt;
        ret += " {\n";
    }

    return ret;
}

std::string buildCheckString([[maybe_unused]] const ForStmt& stmt,
                             const OklLoopInfo& loopInfo,
                             [[maybe_unused]] const TileParams* params,
                             size_t& parenCnt) {
    auto& meta = loopInfo.metadata;
    auto cmpStr = getCondCompStr(meta.condition.op);

    auto ret = util::fmt("if ({} {} {})", meta.var.name, cmpStr, meta.range.end).value();

    if (!isa<CompoundStmt>(stmt.getBody())) {
        ++parenCnt;
        ret += " {\n";
    }

    return ret;
}

HandleResult handleOPENMPTileAttribute(const Attr& a,
                                       const ForStmt& stmt,
                                       const TileParams* params,
                                       SessionStage& s) {
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@tile params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@tile: failed to fetch loop meta data from sema"});
    }
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @tile. Parsed for loop: Init("
                 << "type: " << loopInfo->metadata.var.type
                 << ", name: " << loopInfo->metadata.var.name
                 << ", initValue: " << loopInfo->metadata.range.start
                 << "), Cond(rhsExpr: " << loopInfo->metadata.range.end
                 << "), Inc(rhsInc: " << loopInfo->metadata.inc.val
                 << ", isUnary: " << loopInfo->metadata.isUnary() << ")\n";
#endif

    auto& rewriter = s.getRewriter();
    auto opts = Rewriter::RewriteOptions();
    opts.RemoveLineIfEmpty = true;

    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.RemoveText(attrRange, opts);
    rewriter.RemoveText(SourceRange{stmt.getForLoc(), stmt.getRParenLoc()});

    auto parent = loopInfo->getAttributedParent();

    // Top level `@outer` loop
    if (!parent && loopInfo->metadata.isOuter()) {
        rewriter.InsertText(attrRange.getBegin(), prefixText, false, true);
    }

    size_t parenCnt = 0;

    // First loop. usually `@outer`
    auto firstLoop = buildFirstLoopString(stmt, *loopInfo, params, parenCnt);
    rewriter.InsertText(stmt.getBeginLoc(), firstLoop, false, true);

    // Second loop. usually `@inner`
    auto secondLoop = buildSecondLoopString(stmt, *loopInfo, params, parenCnt);
    rewriter.InsertText(stmt.getRParenLoc().getLocWithOffset(1), secondLoop, true, true);

    // Check code
    if (params->check) {
        auto checkCode = buildCheckString(stmt, *loopInfo, params, parenCnt);
        rewriter.InsertText(stmt.getRParenLoc().getLocWithOffset(1), checkCode, true, true);
    }

    std::string suffixCode = getScopesCloseStr(parenCnt);
    rewriter.InsertText(stmt.getEndLoc(), suffixCode, true, true);

    return {};
}

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, TILE_ATTR_NAME}, makeSpecificAttrHandle(handleOPENMPTileAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << TILE_ATTR_NAME << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
