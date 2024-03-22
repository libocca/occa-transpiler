#include "core/attribute_manager/attribute_store.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>

namespace oklt {
using namespace clang;
using namespace llvm;

void AttributeStore::clear() {
    for (auto& map : _attrMap) {
        map.clear();
    }
}

llvm::SmallVector<AttributeStore::AttrMap*, N_ATTR_KIND> AttributeStore::_selectMap(
    AttributeKind K) {
    if (K != AttributeKind::_ALL) {
        return {&_attrMap[size_t(K)]};
    }

    llvm::SmallVector<AttributeStore::AttrMap*, N_ATTR_KIND> ret = {};
    for (auto it = _attrMap.begin(), end = _attrMap.end(); it != end; ++it) {
        ret.push_back(&(*it));
    }
    return ret;
}

bool AttributeStore::_add(const DynTypedNode& key, Attr& attr, AttributeKind K) {
    assert(K != AttributeKind::_ALL);

    if (!isa<AnnotateAttr, AnnotateTypeAttr, SuppressAttr>(attr)) {
        return false;
    }

    auto& m = *_selectMap(K)[0];
    auto& s = m.getOrInsertDefault(key);

    s.push_back(&attr);
    return true;
}

bool AttributeStore::_add(const QualType& qt, Attr& attr, AttributeKind K) {
    assert(K != AttributeKind::_ALL);

    const auto* t = qt.getTypePtrOrNull();
    if (!t || t->getTypeClass() != clang::Type::Attributed) {
        return false;
    }

    auto node = DynTypedNode::create(*t);
    return _add(node, attr, K);
}

AttrVec AttributeStore::_get(const DynTypedNode& key, AttributeKind K) {
    AttrVec ret = {};
    for (auto m : _selectMap(K)) {
        auto it = m->find(key);
        if (it == m->end()) {
            continue;
        }

        ret.append(it->getSecond());
    }

    return ret;
}

AttrVec AttributeStore::_get(const QualType& qt, AttributeKind K) {
    AttrVec ret = {};

    auto cur = qt, par = cur;
    do {
        auto* t = cur.getTypePtrOrNull();
        if (!t) {
            break;
        }

        if (t->getTypeClass() == clang::Type::Attributed) {
            auto node = DynTypedNode::create(*t);
            for (auto map : _selectMap(K)) {
                auto it = map->find(node);
                if (it == map->end()) {
                    continue;
                }

                ret.append(it->getSecond());
            }
        }

        par = cur;

        if (auto aqt = dyn_cast_or_null<clang::ArrayType>(par)) {
            cur = aqt->getElementType();
            continue;
        }

        cur = par.getSingleStepDesugaredType(_ctx);
    } while (par != cur);

    return ret;
}

bool AttributeStore::_has(const DynTypedNode& node,
                          const SmallVector<StringRef>& ids,
                          AttributeKind K) {
    for (auto map : _selectMap(K)) {
        auto it = map->find(node);
        if (it == map->end()) {
            continue;
        }

        for (auto* attr : it->second) {
            if (!attr) {
                continue;
            }

            auto isAttrMatch = std::any_of(ids.begin(), ids.end(), [attr](auto name) {
                return attr->getNormalizedFullName() == name;
            });
            if (isAttrMatch) {
                return true;
            }
        }
    }

    return false;
}

bool AttributeStore::_has(const QualType& qt, const SmallVector<StringRef>& ids, AttributeKind K) {
    auto cur = qt, par = cur;
    do {
        auto* t = cur.getTypePtrOrNull();
        if (!t) {
            break;
        }

        if (t->getTypeClass() == clang::Type::Attributed) {
            auto node = DynTypedNode::create(*t);
            if (_has(node, ids, K)) {
                return true;
            }
        }

        par = cur;

        if (auto aqt = dyn_cast_or_null<clang::ArrayType>(par)) {
            cur = aqt->getElementType();
            continue;
        }

        cur = par.getSingleStepDesugaredType(_ctx);
    } while (par != cur);

    return false;
}

}  // namespace oklt
