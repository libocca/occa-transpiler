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
    OklLoopInfo* getFirstAttributedChild();

    /* Distance to the for loop tree leave */
    size_t getHeight();
    /* Distance to the for loop tree leave, ignoring loops of other types */
    size_t getHeightSameType();
    size_t getHeightSameType(const LoopMetaType&);

    std::optional<size_t> getSize();

    [[nodiscard]] bool isOuter() const { return metadata.type == LoopMetaType::Outer; };
    [[nodiscard]] bool isInner() const { return metadata.type == LoopMetaType::Inner; };
    [[nodiscard]] bool isOuterInner() const { return metadata.type == LoopMetaType::OuterInner; };
    [[nodiscard]] bool hasOuter() const { return isOuter() || isOuterInner(); };
    [[nodiscard]] bool hasInner() const { return isInner() || isOuterInner(); };
    [[nodiscard]] bool isRegular() const { return metadata.type == LoopMetaType::Regular; };
};

struct OklKernelInfo {
    explicit OklKernelInfo(const clang::FunctionDecl& decl)
        : decl(std::ref(decl)){};
    const std::reference_wrapper<const clang::FunctionDecl> decl;
    std::list<OklLoopInfo> children = {};
};
}  // namespace oklt
