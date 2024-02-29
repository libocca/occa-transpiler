#include "attributes/utils/code_gen.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"

namespace oklt {
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
    // Close all opened scopes
    while (openedScopeCounter--) {
        res += "}";
    }
    return res;
}

HandleResult replaceAttributedLoop(const clang::Attr& a,
                                   const clang::ForStmt& f,
                                   const std::string& prefixCode,
                                   const std::string& suffixCode,
                                   SessionStage& s) {
    auto& rewriter = s.getRewriter();

    // Remove attribute + for loop:
    //      @attribute(...) for (int i = start; i < end; i += inc)
    //  or: for (int i = start; i < end; i += inc; @attribute(...))
    clang::SourceRange range;
    range.setBegin(a.getRange().getBegin().getLocWithOffset(-2));  // TODO: remove magic number
    range.setEnd(f.getRParenLoc());

    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 2)
        .addReplacement(OKL_LOOP_PROLOGUE, range, prefixCode)
        .addReplacement(OKL_LOOP_EPILOGUE, f.getEndLoc(), suffixCode)
        .build();
}

}  // namespace oklt
