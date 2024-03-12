#include "attributes/utils/code_gen.h"
#include "attributes/utils/serial_subset/common.h"

#include <oklt/util/string_utils.h>

#include <clang/Rewrite/Core/Rewriter.h>

namespace oklt::serial_subset {
using namespace clang;

namespace {
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

    // Do not include `for` statement as it's already present.
    //"for ({} {} = {}; {} {} {}; {} {} {})",
    auto ret = util::fmt(" ({} {} = {}; {} {} {}; {} {} {})",
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

}  // namespace

HandleResult handleTileAttribute(const Attr& a,
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

    auto& backendCtx = getBackendCtxFromStage(s);
    auto& rewriter = s.getRewriter();

    SourceRange attr_range = getAttrFullSourceRange(a);
    s.getRewriter().RemoveText(attr_range);

    auto parent = loopInfo->getAttributedParent();

    // `@inner` loop just after `@outer`
    // Top most `@inner` loop
    if (parent && parent->hasOuter() && loopInfo->isInner()) {
        auto& loopInfoEx = backendCtx.getLoopInfo(parent);
        if (!loopInfoEx.exclusive.empty()) {
            s.getRewriter().InsertText(stmt.getBeginLoc(), exclusiveNullText, false, true);
        }
    }

    size_t parenCnt = 0;
    std::string prefixCode;

    // First loop. usually `@outer`
    prefixCode += buildFirstLoopString(stmt, *loopInfo, params, parenCnt);

    // `@inner` loop just after `@outer`
    // Top most `@inner` loop
    if (parent && loopInfo->isOuterInner()) {
        auto& loopInfoEx = backendCtx.getLoopInfo(parent);
        if (!loopInfoEx.exclusive.empty()) {
            prefixCode += (!prefixCode.empty() ? "\n" : "") + exclusiveNullText;
        }
    }

    // Second loop. usually `@inner`
    prefixCode += buildSecondLoopString(stmt, *loopInfo, params, parenCnt);

    // Check code
    if (params->check) {
        prefixCode += buildCheckString(stmt, *loopInfo, params, parenCnt);
    }

    // Replace `for` statement body from LParent to RParen.
    // It is done to avoid replacing the already modified body with insertions before/after.
    rewriter.ReplaceText(SourceRange{stmt.getLParenLoc(), stmt.getRParenLoc()}, prefixCode);

    // Bottom most `@inner` loop
    if (loopInfo->children.empty()) {
        while (parent && !parent->hasOuter()) {
            parent = parent->parent;
        }

        if (parent && !backendCtx.getLoopInfo(parent).exclusive.empty()) {
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
