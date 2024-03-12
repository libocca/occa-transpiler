#pragma once

#include <oklt/core/kernel_metadata.h>

#include <clang/AST/AST.h>

#include <optional>

namespace oklt {

struct KernelInfo;

struct OklLoopInfo {
    const clang::Attr& attr;
    const clang::ForStmt& stmt;
    LoopMetaData& metadata;

    OklLoopInfo* parent = nullptr;
    std::list<OklLoopInfo> children = {};

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
