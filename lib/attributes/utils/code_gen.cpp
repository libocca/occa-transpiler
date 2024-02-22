#include "attributes/utils/code_gen.h"

namespace oklt {
std::string getCondCompStr(const clang::BinaryOperator::Opcode& bo) {
    switch (bo) {
        case clang::BO_LE:
            return "<=";
        case clang::BO_LT:
            return "<";
        case clang::BO_GE:
            return ">=";
        case clang::BO_GT:
            return ">";
        default:  // Shouldn't happen, since for loop parse validates operator
            return "<error>";
    }
}

std::string getUnaryStr(const clang::UnaryOperator::Opcode& uo, const std::string& var) {
    switch (uo) {
        case clang::UO_PreInc:
            return "++" + var;
        case clang::UO_PostInc:
            return var + "++";
        case clang::UO_PreDec:
            return "--" + var;
        case clang::UO_PostDec:
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
