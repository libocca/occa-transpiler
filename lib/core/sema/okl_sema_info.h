#include <oklt/core/kernel_metadata.h>

#include <clang/AST/AST.h>

#include <optional>
#include "attributes/frontend/params/loop.h"

namespace oklt {

struct KernelInfo;

struct OklLoopInfo {
    const clang::Attr& attr;
    const clang::ForStmt& stmt;
    LoopMetaData& metadata;

    OklLoopInfo* parent = nullptr;
    std::list<OklLoopInfo> children = {};

    OklLoopInfo* getAttributedParent();
    OklLoopInfo* getFirstAttributedChild();

    /* Distance to the for loop tree leave */
    size_t getHeight();
    /* Distance to the for loop tree leave, ignoring loops of other types */
    size_t getHeightSameType(const AttributedLoopType&);

    std::optional<size_t> getSize();

    [[nodiscard]] bool isOuter() const;
    [[nodiscard]] bool isInner() const;
    [[nodiscard]] bool isTiled() const;

    [[nodiscard]] bool isOuterInner() const;
    [[nodiscard]] bool isInnerInner() const;
    [[nodiscard]] bool isOuterOuter() const;
    [[nodiscard]] bool isOuterRegular() const;
    [[nodiscard]] bool isInnerRegular() const;
    [[nodiscard]] bool isRegular() const;

    [[nodiscard]] bool hasOuter() const;
    [[nodiscard]] bool hasInner() const;
};

struct OklKernelInfo {
    explicit OklKernelInfo(const clang::FunctionDecl& decl)
        : decl(std::ref(decl)){};
    const std::reference_wrapper<const clang::FunctionDecl> decl;
    std::list<OklLoopInfo> children = {};
};
}  // namespace oklt
