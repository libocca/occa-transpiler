#include "attributes/utils/code_gen.h"

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

}  // namespace oklt
