#pragma once

#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <string>

namespace oklt {
struct LoopMetadata {
    std::string type;
    std::string name;
    struct {
        std::string start;
        std::string end;
        size_t size = 0;
    } range;
    struct {
        std::string cmp;
        clang::BinaryOperator::Opcode op = clang::BO_EQ;
    } condition;
    struct {
        std::string val;
        union {
            clang::UnaryOperator::Opcode uo;
            clang::BinaryOperator::Opcode bo;
        } op;
    } inc;

    bool IsInc() const {
        bool ret = false;
        if (inc.val.empty()) {
            ret = (inc.op.uo == clang::UO_PreInc || inc.op.uo == clang::UO_PostInc);
        } else {
            ret = (inc.op.bo == clang::BO_AddAssign);
        }
        ret = (ret && (condition.op == clang::BO_LE || condition.op == clang::BO_LT));

        return ret;
    };
    std::string getRangeSizeStr() const {
        if (IsInc()) {
            return range.end + " - " + range.start;
        } else {
            return range.start + " - " + range.end;
        };
    };

    bool isUnary() const {
        if (!inc.val.empty()) {
            return false;
        }
        // should by unnecessary check, but just in case
        return (inc.op.uo == clang::UO_PreInc) || (inc.op.uo == clang::UO_PostInc) ||
               (inc.op.uo == clang::UO_PreDec) || (inc.op.uo == clang::UO_PostDec);
    }
};

std::string prettyPrint(const clang::Stmt* S, const clang::PrintingPolicy& policy);
bool EvaluateAsSizeT(const clang::Expr* E, llvm::APSInt& Into, clang::ASTContext& ctx);
LoopMetadata ParseForStmt(clang::ForStmt* S, clang::ASTContext& ctx);

}  // namespace oklt
