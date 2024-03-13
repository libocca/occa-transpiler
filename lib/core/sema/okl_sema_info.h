#pragma once

#include <oklt/core/kernel_metadata.h>

#include <clang/AST/AST.h>

#include <optional>
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

namespace oklt {

struct KernelInfo;

using AttributedLoopTypes = std::vector<LoopType>;

struct OklLoopInfo {
    const clang::Attr& attr;
    const clang::ForStmt& stmt;
    AttributedLoopTypes type = {LoopType::Regular};

    OklLoopInfo* parent = nullptr;
    std::list<OklLoopInfo> children = {};

    struct {
        std::string typeName;
        std::string name;
        const clang::VarDecl* varDecl;
    } var;
    struct {
        std::string start;
        std::string end;
        const clang::Expr* start_;
        const clang::Expr* end_;
        size_t size = 0;
    } range;
    struct {
        const clang::BinaryOperator* cmp_;
        BinOp op = BinOp::Eq;
    } condition;
    struct {
        std::string val;
        const clang::Expr* val_;
        union {
            UnOp uo;
            BinOp bo;
        } op;
    } inc;

    [[nodiscard]] bool IsInc() const {
        bool ret = false;
        if (inc.val.empty()) {
            ret = (inc.op.uo == UnOp::PreInc || inc.op.uo == UnOp::PostInc);
        } else {
            ret = (inc.op.bo == BinOp::AddAssign);
        }

        ret = (ret && (condition.op == BinOp::Le || condition.op == BinOp::Lt));

        return ret;
    };
    [[nodiscard]] bool isUnary() const {
        if (!inc.val.empty()) {
            return false;
        }
        // should by unnecessary check, but just in case
        return (inc.op.uo == UnOp::PreInc) || (inc.op.uo == UnOp::PostInc) ||
               (inc.op.uo == UnOp::PreDec) || (inc.op.uo == UnOp::PostDec);
    };

    [[nodiscard]] std::string getRangeSizeStr() const {
        if (IsInc()) {
            return range.end + " - " + range.start;
        } else {
            return range.start + " - " + range.end;
        };
    };

    OklLoopInfo* getAttributedParent();
    OklLoopInfo* getAttributedParent(std::function<bool(OklLoopInfo&)> f);
    OklLoopInfo* getFirstAttributedChild();
    OklLoopInfo* getFirstAttributedChild(std::function<bool(OklLoopInfo&)> f);

    /* Distance to the for loop tree leave */
    size_t getHeight();
    /* Distance to the for loop tree leave, ignoring loops of other types */
    size_t getHeightSameType(const LoopType&);

    std::optional<size_t> getSize();

    [[nodiscard]] bool is(const LoopType&) const;
    [[nodiscard]] bool is(const LoopType&, const LoopType&) const;
    [[nodiscard]] bool has(const LoopType&) const;
    [[nodiscard]] bool isTiled() const;
    [[nodiscard]] bool isRegular() const;
};

struct OklKernelInfo {
    explicit OklKernelInfo(const clang::FunctionDecl& decl)
        : decl(std::ref(decl)){};
    const std::reference_wrapper<const clang::FunctionDecl> decl;
    std::list<OklLoopInfo> children = {};
};
}  // namespace oklt
