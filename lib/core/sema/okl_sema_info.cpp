#include "core/sema//okl_sema_info.h"

#include <deque>
#include "oklt/core/kernel_metadata.h"

namespace oklt {

OklLoopInfo* OklLoopInfo::getAttributedParent() {
    auto ret = parent;
    while (ret && ret->isRegular()) {
        ret = ret->parent;
    }
    return ret;
}

OklLoopInfo* OklLoopInfo::getFirstAttributedChild() {
    std::deque<OklLoopInfo*> elements = {};
    for (auto& v : children) {
        elements.push_back(&v);
    }

    while (!elements.empty()) {
        auto el = elements.front();
        elements.pop_front();
        if (!el->isRegular())
            return el;

        for (auto& v : el->children) {
            elements.push_back(&v);
        }
    }

    return nullptr;
}

std::optional<size_t> OklLoopInfo::getSize() {
    if (isRegular()) {
        if (children.empty()) {
            return std::nullopt;
        }
    } else if (metadata.range.size == 0) {
        return std::nullopt;
    }

    auto sz = size_t{1};
    sz = std::max(metadata.range.size, sz);

    auto ret = sz;
    for (auto& child : children) {
        auto v = child.getSize();
        if (!v.has_value()) {
            return std::nullopt;
        }
        ret = std::max(sz * v.value(), ret);
    }
    return ret;
}

size_t OklLoopInfo::getHeight() {
    OklLoopInfo* currLoop = this;
    int h = 0;
    while (!currLoop->children.empty()) {
        currLoop = currLoop->getFirstAttributedChild();
        ++h;
    }
    return h;
}

size_t OklLoopInfo::getHeightSameType() {
    return getHeightSameType(metadata.type);
}

size_t OklLoopInfo::getHeightSameType(const LoopMetaType& type) {
    OklLoopInfo* currLoop = this;
    int h = 0;
    while (!currLoop->children.empty()) {
        currLoop = currLoop->getFirstAttributedChild();
        if (currLoop->metadata.type == type ||
            (currLoop->metadata.type == LoopMetaType::OuterInner &&
             (type == LoopMetaType::Outer || type == LoopMetaType::Inner))) {
            ++h;
        }
    }
    return h;
}

}  // namespace oklt
