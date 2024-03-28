#include "attributes/frontend/params/tile.h"
#include "attributes/utils/code_gen.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"
#include "oklt/core/kernel_metadata.h"

#include <oklt/util/string_utils.h>

#include <clang/Rewrite/Core/Rewriter.h>

#include <spdlog/spdlog.h>
namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string exclusiveNullText = "_occa_exclusive_index = 0;\n";
const std::string exclusiveIncText = "++_occa_exclusive_index;\n";

std::string getTiledVariableName(const OklLoopInfo& forLoop) {
    return "_occa_tiled_" + forLoop.var.name;
}

std::string getScopesCloseStr(size_t& parenCnt) {
    std::string ret;
    while (parenCnt--) {
        ret += "}\n";
    }
    return ret;
}

std::string buildFirstLoopString([[maybe_unused]] const ForStmt& stmt,
                                 const OklLoopInfo& forLoop,
                                 [[maybe_unused]] const TileParams* params,
                                 size_t& parenCnt,
                                 clang::Rewriter& rewriter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(forLoop.condition.op);
    auto incValStr = params->tileSize;
    if (forLoop.inc.val) {
        incValStr =
            util::fmt("({} * {})", params->tileSize, getLatestSourceText(forLoop.inc.val, rewriter))
                .value();
    }

    // Do not include `for` statement as it's already present.
    //"for ({} {} = {}; {} {} {}; {} {} {})",
    auto ret = util::fmt(" ({} {} = ({}); {} {} {}; {} {} {})",
                         forLoop.var.typeName,
                         tiledVar,
                         getLatestSourceText(forLoop.range.start, rewriter),
                         tiledVar,
                         cmpOpStr,
                         getLatestSourceText(forLoop.range.end, rewriter),
                         tiledVar,
                         assignUpdate,
                         incValStr)
                   .value();

    ++parenCnt;
    ret += " {\n";

    return ret;
}

std::string buildSecondLoopString(const ForStmt& stmt,
                                  const OklLoopInfo& forLoop,
                                  const TileParams* params,
                                  size_t& parenCnt,
                                  clang::Rewriter& rewriter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto op = forLoop.IsInc() ? "+" : "-";
    auto cmp = forLoop.IsInc() ? "<" : ">";

    std::string ret;
    if (forLoop.isUnary()) {
        auto unaryStr = getUnaryStr(forLoop.inc.op.uo, forLoop.var.name);
        ret = util::fmt("for ({} {} = {}; {} {} ({} {} {}); {})",
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
        ret = util::fmt("for ({} {} = {}; {} {} ({} {} {}); {} {} {})",
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

    if (params->check || !isa<CompoundStmt>(stmt.getBody())) {
        ++parenCnt;
        ret += " {\n";
    }

    return ret;
}

std::string buildCheckString(const ForStmt& stmt,
                             const OklLoopInfo& forLoop,
                             [[maybe_unused]] const TileParams* params,
                             size_t& parenCnt,
                             clang::Rewriter& rewriter) {
    auto cmpStr = getCondCompStr(forLoop.condition.op);

    auto ret = util::fmt("if ({} {} {})",
                         forLoop.var.name,
                         cmpStr,
                         getLatestSourceText(forLoop.range.end, rewriter))
                   .value();

    if (!isa<CompoundStmt>(stmt.getBody())) {
        ++parenCnt;
        ret += " {\n";
    }

    return ret;
}

}  // namespace

HandleResult handleTileAttribute(const Attr& a,
                                 const ForStmt& stmt,
                                 const TileParams* params,
                                 SessionStage& s) {
    SPDLOG_DEBUG("Handle [@tile] attribute");

    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@tile params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@tile: failed to fetch loop meta data from sema"});
    }

    removeAttribute(a, s);

    auto& rewriter = s.getRewriter();

    SourceRange attr_range = getAttrFullSourceRange(a);
    s.getRewriter().RemoveText(attr_range);

    auto parent = loopInfo->getAttributedParent();

    // `@inner` loop just after `@outer`
    // Top most `@inner` loop
    if (parent && parent->has(LoopType::Outer) && loopInfo->is(LoopType::Inner)) {
        if (parent->exclusiveInfo.declared) {
            s.getRewriter().InsertText(stmt.getBeginLoc(), exclusiveNullText, false, true);
        }
    }

    size_t parenCnt = 0;
    std::string prefixCode;

    // First loop. usually `@outer`
    prefixCode += buildFirstLoopString(stmt, *loopInfo, params, parenCnt, s.getRewriter());

    // `@inner` loop just after `@outer`
    // Top most `@inner` loop
    if (parent && loopInfo->is(LoopType::Outer, LoopType::Inner)) {
        if (loopInfo->exclusiveInfo.declared) {
            prefixCode += (!prefixCode.empty() ? "\n" : "") + exclusiveNullText;
        }
    }

    // Second loop. usually `@inner`
    prefixCode += buildSecondLoopString(stmt, *loopInfo, params, parenCnt, s.getRewriter());

    // Check code
    if (params->check) {
        prefixCode += buildCheckString(stmt, *loopInfo, params, parenCnt, s.getRewriter());
    }

    // Replace `for` statement body from LParent to RParen.
    // It is done to avoid replacing the already modified body with insertions before/after.
    rewriter.ReplaceText(SourceRange{stmt.getLParenLoc(), stmt.getRParenLoc()}, prefixCode);

    // Bottom most `@inner` loop
    if (loopInfo->children.empty()) {
        auto outerParent =
            loopInfo->getAttributedParent([](OklLoopInfo& v) { return v.has(LoopType::Outer); });
        if (outerParent && outerParent->exclusiveInfo.used) {
            auto compStmt = dyn_cast_or_null<CompoundStmt>(loopInfo->stmt.getBody());
            SourceLocation incLoc =
                compStmt ? compStmt->getRBracLoc().getLocWithOffset(-1) : stmt.getEndLoc();
            rewriter.InsertTextBefore(incLoc, exclusiveIncText);
        }
    }

    std::string suffixCode = getScopesCloseStr(parenCnt);
    rewriter.InsertTextAfter(stmt.getEndLoc(), suffixCode);

    return {};
}

}  // namespace oklt::serial_subset
