#pragma once

#include "../../../../../../../usr/lib/llvm-17/include/clang/AST/ASTTypeTraits.h"
#include "../../../../../../../usr/lib/llvm-17/include/llvm/ADT/DenseSet.h"

namespace oklt {

/// Holds a map of nodes and their custom attributes.
///
/// This class does not call attribute destructors,
/// please take care of their proper destruction by calling `ASTContext::addDestruction(Attr)` after
/// their creation.
class AttributedTypeMap {
   public:
    explicit AttributedTypeMap() = default;

    bool add(const clang::QualType& qt, clang::Attr* attr);
    clang::AttrVec get(clang::ASTContext& ctx, const clang::QualType& qt);
    bool has(clang::ASTContext& ctx,
             const clang::QualType& qt,
             const llvm::SmallVector<clang::StringRef>& ids);

   private:
    llvm::DenseMap<const clang::Type*, clang::Attr*> _attrMap;
};

}  // namespace oklt
