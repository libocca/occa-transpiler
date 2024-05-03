#include "attributed_type_map.h"

#include <clang/AST/Attr.h>

using namespace oklt;
using namespace clang;
using namespace llvm;

bool AttributedTypeMap::add(const QualType& qt, Attr* attr) {
    const auto* key = qt.getTypePtrOrNull();
    if (!key || !isa<AnnotateAttr, AnnotateTypeAttr, SuppressAttr>(attr)) {
        return false;
    }

    auto [_, ret] = _attrMap.try_emplace(key, attr);
    return ret;
}

AttrVec AttributedTypeMap::get(clang::ASTContext& ctx, const QualType& qt) {
    AttrVec ret = {};

    auto cur = qt, par = cur;
    do {
        const auto* key = cur.getTypePtrOrNull();
        auto it = _attrMap.find(key);
        if (it != _attrMap.end()) {
            ret.push_back(it->second);
        }

        par = cur;

        if (auto aqt = dyn_cast_or_null<clang::ArrayType>(par)) {
            cur = aqt->getElementType();
            continue;
        }

        cur = par.getSingleStepDesugaredType(ctx);
    } while (par != cur);

    return ret;
}

bool AttributedTypeMap::has(clang::ASTContext& ctx,
                            const QualType& qt,
                            const SmallVector<StringRef>& ids) {
    auto cur = qt, par = cur;
    do {
        const auto* key = cur.getTypePtrOrNull();
        auto it = _attrMap.find(key);
        if (it != _attrMap.end()) {
            auto* attr = it->second;
            auto isAttrMatch = std::any_of(ids.begin(), ids.end(), [attr](auto name) {
                return attr->getNormalizedFullName() == name;
            });
            if (isAttrMatch) {
                return true;
            }
        }

        par = cur;

        if (auto aqt = dyn_cast_or_null<clang::ArrayType>(par)) {
            cur = aqt->getElementType();
            continue;
        }

        cur = par.getSingleStepDesugaredType(ctx);
    } while (par != cur);

    return false;
}
