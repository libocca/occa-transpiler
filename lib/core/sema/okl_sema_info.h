#pragma once

#include <oklt/core/kernel_metadata.h>

#include <clang/AST/AST.h>

#include <clang/AST/Decl.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Type.h>
#include <optional>

namespace oklt {

struct KernelInfo;

using AttributedLoopTypes = std::vector<LoopType>;

struct OklLoopInfo {
    struct AttributedTypeInfo {
        // variable of this attributed type declared in THIS loop (applied to @outer only)
        bool declared = false;
        // variable of this attributed type used in this or child loops (applied to @inner only)
        bool used = false;
    };

    using OptSize = std::optional<size_t>;
    struct OptSizes: public std::list<OptSize> {
        size_t product();
        bool hasNullOpts();
        bool allNullOpts();
    };

    const clang::Attr& attr;
    const clang::ForStmt& stmt;
    AttributedLoopTypes type = {LoopType::Regular};

    OklLoopInfo* parent = nullptr;
    std::list<OklLoopInfo> children = {};
    std::string tileSize = "";

    AttributedTypeInfo sharedInfo;
    AttributedTypeInfo exclusiveInfo;

    struct {
        std::string typeName;
        std::string name;
        const clang::VarDecl* varDecl;
    } var;
    struct {
        const clang::Expr* start;
        const clang::Expr* end;
        size_t size = 0;
    } range;
    struct {
        const clang::BinaryOperator* cmp;
        BinOp op = BinOp::Eq;
    } condition;
    struct {
        const clang::Expr* val;
        union {
            UnOp uo;
            BinOp bo;
        } op;
    } inc;

    [[nodiscard]] bool shouldSync();
    void markSharedUsed();
    void markExclusiveUsed();

    [[nodiscard]] bool IsInc() const;
    [[nodiscard]] bool isUnary() const;

    OklLoopInfo* getAttributedParent();
    OklLoopInfo* getAttributedParent(std::function<bool(OklLoopInfo&)> f);
    OklLoopInfo* getFirstAttributedChild();
    OklLoopInfo* getFirstAttributedChild(std::function<bool(OklLoopInfo&)> f);

    /* Distance to the for loop tree leave */
    size_t getHeight();
    /* Distance to the for loop tree leave, ignoring loops of other types */
    size_t getHeightSameType(const LoopType&);

    OptSizes getInnerSizes();

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
