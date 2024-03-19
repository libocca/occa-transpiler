#include "core/sema/okl_sema_info.h"

#include <deque>
#include <numeric>
#include <optional>
#include "oklt/core/kernel_metadata.h"

namespace oklt {

[[nodiscard]] bool OklLoopInfo::shouldSync() {
    // 1. There should be shared memory usage somewhere inside loop
    if (!sharedInfo.used) {
        return false;
    }

    // 2. Should be highest @inner loop
    if (!(is(LoopType::Inner) || is(LoopType::Inner, LoopType::Inner))) {
        return false;
    }
    auto* parent = getAttributedParent();
    if (!parent || parent->has(LoopType::Inner)) {
        return false;
    }

    // 3. Should not be the last @inner loop
    if (&parent->children.back() == this) {
        return false;
    }

    return true;
}

void OklLoopInfo::markSharedUsed() {
    // Mark this loop and all of its ancestors
    sharedInfo.used = true;
    auto* curLoop = parent;
    while (curLoop) {
        curLoop->sharedInfo.used = true;
        curLoop = curLoop->parent;
    }
}

void OklLoopInfo::markExclusiveUsed() {
    // Mark this loop and all of its ancestors
    exclusiveInfo.used = true;
    auto* curLoop = parent;
    while (curLoop) {
        curLoop->exclusiveInfo.used = true;
        curLoop = curLoop->parent;
    }
}

[[nodiscard]] bool OklLoopInfo::IsInc() const {
    bool ret = false;
    if (!inc.val) {
        ret = (inc.op.uo == UnOp::PreInc || inc.op.uo == UnOp::PostInc);
    } else {
        ret = (inc.op.bo == BinOp::AddAssign);
    }

    ret = (ret && (condition.op == BinOp::Le || condition.op == BinOp::Lt));

    return ret;
};
[[nodiscard]] bool OklLoopInfo::isUnary() const {
    if (inc.val) {
        return false;
    }
    // should by unnecessary check, but just in case
    return (inc.op.uo == UnOp::PreInc) || (inc.op.uo == UnOp::PostInc) ||
           (inc.op.uo == UnOp::PreDec) || (inc.op.uo == UnOp::PostDec);
};

[[nodiscard]] bool OklLoopInfo::is(const LoopType& loopType) const {
    return type.size() == 1 && type.front() == loopType;
};
[[nodiscard]] bool OklLoopInfo::is(const LoopType& loopType1, const LoopType& loopType2) const {
    return type.size() == 2 && type[0] == loopType1 && type[1] == loopType2;
};

[[nodiscard]] bool OklLoopInfo::isTiled() const {
    return type.size() == 2;
};
[[nodiscard]] bool OklLoopInfo::has(const LoopType& loopType) const {
    for (auto& currLoopType : type) {
        if (currLoopType == loopType) {
            return true;
        }
    }
    return false;
};
[[nodiscard]] bool OklLoopInfo::isRegular() const {
    for (auto& loopType : type) {
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

OklLoopInfo::OptSizes OklLoopInfo::getInnerSizes() {
    if (isRegular()) {
        if (children.empty()) {
            return {};
        }
    } else if (range.size == 0) {
        return OklLoopInfo::OptSizes{{0}};
    }

    OptSize sz = size_t{1};
    if (is(LoopType::Inner)) {
        sz = std::max(range.size, sz.value());
    } else if (is(LoopType::Outer, LoopType::Inner)) {
        if (!tileSize.empty()) {
            // TODO: maybe reuse attribute parser
            char* p;
            auto tileSizeLL = std::strtoll(tileSize.c_str(), &p, 10);
            if (*p) {
                sz = std::nullopt;
            } else {
                sz = static_cast<size_t>(tileSizeLL);
            }
        }
    }

    OklLoopInfo::OptSizes ret;
    size_t prevProd = 0;
    for (auto& child : children) {
        auto currSizes = child.getInnerSizes();
        auto prod = currSizes.product();
        if (prod > prevProd) {
            ret = currSizes;
        }
    }
    if (has(LoopType::Inner)) {
        ret.emplace_front(sz);
    }
    return ret;
}

size_t OklLoopInfo::getHeight() {
    auto* currLoop = this;
    int h = 0;
    while (!currLoop->children.empty()) {
        currLoop = currLoop->getFirstAttributedChild();
        h += currLoop->type.size();
    }
    return h;
}

size_t OklLoopInfo::getHeightSameType(const LoopType& type) {
    auto* currLoop = this;
    int h = 0;
    currLoop = currLoop->getFirstAttributedChild();
    while (currLoop) {
        for (auto& loopType : currLoop->type) {
            if (loopType == type) {
                ++h;
            }
        }
        currLoop = currLoop->getFirstAttributedChild();
    }
    return h;
}

size_t OklLoopInfo::OptSizes::product() {
    return std::accumulate(begin(), end(), size_t{1}, [](const OptSize& a, const OptSize& b) {
        size_t aZ = a.value_or(size_t{1});
        size_t bZ = b.value_or(size_t{1});
        return aZ * bZ;
    });
}
bool OklLoopInfo::OptSizes::hasNullOpts() {
    return std::find_if(begin(), end(), [](const auto& val) { return val == std::nullopt; }) !=
           end();
}

}  // namespace oklt
