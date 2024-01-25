#pragma once

#include <clang/AST/ASTTypeTraits.h>
#include <llvm/ADT/DenseSet.h>

namespace oklt {

class SessionStage;

/// Holds a map of nodes and their custom attributes.

/// This class does not call attribute destructors,
/// please take care of their proper destruction by calling `ASTContext::addDestruction(Attr)` after their creation.
class AttributeStore {
public:
 explicit AttributeStore(SessionStage& _stage);
 ~AttributeStore();

  void add(const clang::DynTypedNode& node, clang::Attr *attr);
  void add(const clang::QualType& qt, clang::Attr *attr);

  clang::AttrVec get(const clang::DynTypedNode& node);
  clang::AttrVec get(const clang::QualType& qt);

  bool has(const clang::QualType& qt, const llvm::SmallVector<clang::StringRef>& ids);
  bool has(const clang::DynTypedNode& node, const llvm::SmallVector<clang::StringRef>& ids);

  void clear();

private:
  clang::ASTContext& _ctx;
  llvm::DenseMap<clang::DynTypedNode, clang::AttrVec> _attrMap;
};

} // namespace oklt
