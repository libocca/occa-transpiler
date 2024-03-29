#include "attributes/utils/code_gen.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/utils/attributes.h"

namespace oklt {
using namespace clang;

std::string getCondCompStr(const BinOp& bo) {
    switch (bo) {
        case BinOp::Le:
            return "<=";
        case BinOp::Lt:
            return "<";
        case BinOp::Ge:
            return ">=";
        case BinOp::Gt:
            return ">";
        default:  // Shouldn't happen, since for loop parse validates operator
            return "<error>";
    }
}

std::string getUnaryStr(const UnOp& uo, const std::string& var) {
    switch (uo) {
        case UnOp::PreInc:
            return "++" + var;
        case UnOp::PostInc:
            return var + "++";
        case UnOp::PreDec:
            return "--" + var;
        case UnOp::PostDec:
            return var + "--";

        default:  // Shouldn't happen, since for loop parse validates operator
            return "<error>";
    }
}

std::string buildCloseScopes(int& openedScopeCounter) {
    std::string res;
    res.reserve(openedScopeCounter * 2);

    // Close all opened scopes
    while (openedScopeCounter--) {
        res += "}\n";
    }

    return res;
}

HandleResult replaceAttributedLoop(const Attr& a,
                                   const ForStmt& f,
                                   const std::string& prefixCode,
                                   const std::string& suffixCode,
                                   SessionStage& s,
                                   bool insertInside) {
    return replaceAttributedLoop(a, f, prefixCode, suffixCode, "", s, insertInside);
}

HandleResult replaceAttributedLoop(const Attr& a,
                                   const ForStmt& f,
                                   const std::string& prefixCode,
                                   const std::string& suffixCode,
                                   const std::string& afterCode,
                                   SessionStage& s,
                                   bool insertInside) {
    auto& rewriter = s.getRewriter();

    rewriter.RemoveText(getAttrFullSourceRange(a));

    if (insertInside) {
        auto body = dyn_cast_or_null<CompoundStmt>(f.getBody());
        if (body) {
            rewriter.RemoveText(SourceRange{f.getForLoc(), f.getRParenLoc()});
            rewriter.InsertText(body->getLBracLoc().getLocWithOffset(1),
                                std::string("\n") + prefixCode,
                                true,
                                true);
            rewriter.InsertText(f.getEndLoc(), suffixCode, true, true);
        } else {
            rewriter.ReplaceText(SourceRange{f.getForLoc(), f.getRParenLoc()},
                                 std::string("{\n") + prefixCode);
            rewriter.InsertTextAfterToken(f.getEndLoc().getLocWithOffset(1), suffixCode + "\n}");
        }
    } else {
        rewriter.ReplaceText(SourceRange{f.getForLoc(), f.getRParenLoc()}, prefixCode);
        rewriter.InsertText(f.getEndLoc(), suffixCode, true, true);
    }

    if (!afterCode.empty()) {
        rewriter.InsertText(f.getEndLoc().getLocWithOffset(1), afterCode);
    }

    return {};
}

}  // namespace oklt
