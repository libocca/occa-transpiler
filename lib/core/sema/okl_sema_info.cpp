#include "core/sema/okl_sema_info.h"

#include <deque>
#include "oklt/core/kernel_metadata.h"

namespace oklt {

[[nodiscard]] bool OklLoopInfo::is(const LoopType& loopType) const {
    return metadata.type.size() == 1 && metadata.type.front() == loopType;
};
[[nodiscard]] bool OklLoopInfo::is(const LoopType& loopType1, const LoopType& loopType2) const {
    return metadata.type.size() == 2 && metadata.type[0] == loopType1 &&
           metadata.type[1] == loopType2;
};

[[nodiscard]] bool OklLoopInfo::isTiled() const {
    return metadata.type.size() == 2;
};
[[nodiscard]] bool OklLoopInfo::has(const LoopType& loopType) const {
    for (auto& currLoopType : metadata.type) {
        if (currLoopType == loopType) {
            return true;
        }
    }
    return false;
};
[[nodiscard]] bool OklLoopInfo::isRegular() const {
    for (auto& loopType : metadata.type) {
        if (loopType != LoopType::Regular) {
            return false;
        }
    }
    return true;
};

OklLoopInfo* OklLoopInfo::getAttributedParent() {
    auto ret = parent;
    while (ret && ret->isRegular()) {
        ret = ret->parent;
    }
    return ret;
}

OklLoopInfo* OklLoopInfo::getAttributedParent(std::function<bool(OklLoopInfo&)> f) {
    auto ret = parent;
    while (ret && !f(*ret)) {
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

OklLoopInfo* OklLoopInfo::getFirstAttributedChild(std::function<bool(OklLoopInfo&)> f) {
    std::deque<OklLoopInfo*> elements = {};
    for (auto& v : children) {
        elements.push_back(&v);
    }

    while (!elements.empty()) {
        auto el = elements.front();
        elements.pop_front();
        if (f(*el))
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
        h += currLoop->metadata.type.size();
    }
    return h;
}

size_t OklLoopInfo::getHeightSameType(const LoopType& type) {
    OklLoopInfo* currLoop = this;
    int h = 0;
    while (!currLoop->children.empty()) {
        currLoop = currLoop->getFirstAttributedChild();
        for (auto& loopType : currLoop->metadata.type) {
            if (loopType == type) {
                ++h;
            }
        }
    }
    return h;
}

}  // namespace oklt
