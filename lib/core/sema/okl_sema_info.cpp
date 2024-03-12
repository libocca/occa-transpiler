#include "core/sema//okl_sema_info.h"

#include <deque>
#include "attributes/frontend/params/loop.h"
#include "oklt/core/kernel_metadata.h"

namespace oklt {

[[nodiscard]] bool OklLoopInfo::isOuter() const {
    return metadata.type.size() == 1 && metadata.type.front() == AttributedLoopType::Outer;
};
[[nodiscard]] bool OklLoopInfo::isInner() const {
    return metadata.type.size() == 1 && metadata.type.front() == AttributedLoopType::Inner;
};

[[nodiscard]] bool OklLoopInfo::isOuterInner() const {
    return isTiled() && metadata.type[0] == AttributedLoopType::Outer &&
           metadata.type[1] == AttributedLoopType::Inner;
};

[[nodiscard]] bool OklLoopInfo::isInnerInner() const {
    return isTiled() && metadata.type[0] == AttributedLoopType::Inner &&
           metadata.type[1] == AttributedLoopType::Inner;
};

[[nodiscard]] bool OklLoopInfo::isOuterOuter() const {
    return isTiled() && metadata.type[0] == AttributedLoopType::Outer &&
           metadata.type[1] == AttributedLoopType::Outer;
};

[[nodiscard]] bool OklLoopInfo::isOuterRegular() const {
    return isTiled() && metadata.type[0] == AttributedLoopType::Outer &&
           metadata.type[1] == AttributedLoopType::Regular;
};

[[nodiscard]] bool OklLoopInfo::isInnerRegular() const {
    return isTiled() && metadata.type[0] == AttributedLoopType::Inner &&
           metadata.type[1] == AttributedLoopType::Regular;
};

[[nodiscard]] bool OklLoopInfo::isTiled() const {
    return metadata.type.size() == 2;
};
[[nodiscard]] bool OklLoopInfo::hasOuter() const {
    for (auto& loopType : metadata.type) {
        if (loopType == AttributedLoopType::Outer) {
            return true;
        }
    }
    return false;
};
[[nodiscard]] bool OklLoopInfo::hasInner() const {
    for (auto& loopType : metadata.type) {
        if (loopType == AttributedLoopType::Inner) {
            return true;
        }
    }
    return false;
};
[[nodiscard]] bool OklLoopInfo::isRegular() const {
    for (auto& loopType : metadata.type) {
        if (loopType != AttributedLoopType::Regular) {
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
        h += currLoop->metadata.type.size();
    }
    return h;
}

size_t OklLoopInfo::getHeightSameType(const AttributedLoopType& type) {
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
