#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/AST/ASTTypeTraits.h>

namespace oklt {

enum AttributeKind { COMMON, MODIFIER, _ALL };
constexpr auto MAX_KIND_TYPE_SZ = static_cast<size_t>(AttributeKind::MODIFIER);
constexpr auto N_ATTR_KIND = MAX_KIND_TYPE_SZ + 1;

/// Holds a map of nodes and their custom attributes.
///
/// This class does not call attribute destructors,
/// please take care of their proper destruction by calling `ASTContext::addDestruction(Attr)` after
/// their creation.
class AttributeStore {
   public:
    explicit AttributeStore(clang::ASTContext& ctx)
        : _ctx(ctx){};

    template <typename T>
    bool add(const T& key, clang::Attr& attr, AttributeKind attrKind = AttributeKind::COMMON) {
        if constexpr (std::is_same_v<clang::QualType, T>) {
            return _add(key, attr, attrKind);
        }

        auto node = clang::DynTypedNode::create(key);
        return _add(node, attr, attrKind);
    };

    template <typename T>
    clang::AttrVec get(const T& key, AttributeKind attrKind = AttributeKind::_ALL) {
        if constexpr (std::is_same_v<clang::QualType, T>) {
            return _get(key, attrKind);
        }

        auto node = clang::DynTypedNode::create(key);
        return _get(node, attrKind);
    };

    template <typename T>
    bool has(const T& key,
             const llvm::SmallVector<clang::StringRef>& ids,
             AttributeKind attrKind = AttributeKind::_ALL) {
        if constexpr (std::is_same_v<clang::QualType, T>) {
            return _has(key, ids, attrKind);
        }

        auto node = clang::DynTypedNode::create(key);
        return _has(node, ids, attrKind);
    };

    void clear();

   private:
    using AttrMap = llvm::DenseMap<clang::DynTypedNode, clang::AttrVec>;

    clang::ASTContext& _ctx;
    std::array<AttrMap, N_ATTR_KIND> _attrMap;

    llvm::SmallVector<AttributeStore::AttrMap*, N_ATTR_KIND> _selectMap(AttributeKind);

    bool _add(const clang::DynTypedNode&, clang::Attr&, AttributeKind);
    bool _add(const clang::QualType&, clang::Attr&, AttributeKind);

    clang::AttrVec _get(const clang::DynTypedNode&, AttributeKind);
    clang::AttrVec _get(const clang::QualType&, AttributeKind);

    bool _has(const clang::DynTypedNode&,
              const llvm::SmallVector<clang::StringRef>&,
              AttributeKind);
    bool _has(const clang::QualType&, const llvm::SmallVector<clang::StringRef>&, AttributeKind);
};

}  // namespace oklt
