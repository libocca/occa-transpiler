#include "oklt/core/attribute_manager/attribute_store.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>

using namespace oklt;
using namespace clang;
using namespace llvm;

AttributeStore::AttributeStore(ASTContext& ctx): _ctx(ctx) {};

void AttributeStore::add(const DynTypedNode& node, Attr *attr) {
  if (!attr || !isa<AnnotateAttr, AnnotateTypeAttr, SuppressAttr>(attr))
    return;

  auto [it, _] = _attrMap.try_emplace(node, AttrVec());
  it->getSecond().push_back(attr);
}

void AttributeStore::add(const QualType& qt, Attr *attr) {
  add(clang::DynTypedNode::create(*qt.getTypePtrOrNull()), attr);
}

static AttrVec emptyVec = {};
AttrVec AttributeStore::get(const DynTypedNode& node) {
  auto it = _attrMap.find(node);
  if (it == _attrMap.end())
    return emptyVec;

  return it->getSecond();
}

AttrVec AttributeStore::get(const QualType& qt) {
  AttrVec ret = {};

  auto cur = qt, par = cur;
  do {
    ret.append(get(DynTypedNode::create(*cur.getTypePtrOrNull())));

    par = cur;
    cur = par.getSingleStepDesugaredType(_ctx);
  } while (par != cur);

  return ret;
}

bool AttributeStore::has(const DynTypedNode& node, const SmallVector<StringRef> ids) {
  for (auto attr: get(node)) {
    if (std::find(ids.begin(), ids.end(), attr->getNormalizedFullName()) != ids.end())
      return true;
  }

  return false;
}

bool AttributeStore::has(const QualType& qt, const SmallVector<StringRef> ids) {
  AttrVec ret = {};

  auto cur = qt, par = cur;
  do {
    if (has(DynTypedNode::create(*cur.getTypePtrOrNull()), ids))
      return true;

    par = cur;
    cur = par.getSingleStepDesugaredType(_ctx);
  } while (par != cur);

  return false;
}
