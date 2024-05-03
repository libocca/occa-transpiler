#include "core/sema/okl_sema_info.h"

#include <deque>
#include <numeric>
#include <optional>
#include "attributes/frontend/params/loop.h"
#include "oklt/core/kernel_metadata.h"

namespace oklt {

// Original OCCA takes only first loops branch. Correct implementation should find the biggest
//  sizes among all branches
// #define LEGACY_INNER_SIZES_CALCULATION

[[nodiscard]] bool OklLoopInfo::shouldSync() {
    // 1. There should be shared memory usage somewhere inside loop
    if (!sharedInfo.used || sharedInfo.nobarrierApplied) {
        return false;
    }

    // 2. Should be highest @inner loop
    if (!(is(LoopType::Inner) || is(LoopType::Inner, LoopType::Inner))) {
        return false;
    }
    auto* p = parent;
    if (!p || p->has(LoopType::Inner)) {
        return false;
    }

    // 3. Should not be the last @inner loop (if parent is Outer)
    if ((p->type.back() == LoopType::Outer) && &p->children.back() == this) {
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

[[nodiscard]] bool OklLoopInfo::has(const LoopType& loopType) const {
    for (auto& currLoopType : type) {
        if (currLoopType == loopType) {
            return true;
        }
    }
    return false;
};

[[nodiscard]] bool OklLoopInfo::isTiled() const {
    return type.size() == 2;
};

[[nodiscard]] bool OklLoopInfo::isRegular() const {
    for (auto& loopType : type) {
        if (loopType != LoopType::Regular) {
            return false;
        }
    }
    return true;
};

[[nodiscard]] bool OklLoopInfo::is(const Axis& loopAxis) const {
    return type.size() == 1 && axis.front() == loopAxis;
};
[[nodiscard]] bool OklLoopInfo::is(const Axis& loopAxis1, const Axis& loopAxis2) const {
    return type.size() == 2 && axis[0] == loopAxis1 && axis[1] == loopAxis2;
};

[[nodiscard]] bool OklLoopInfo::has(const Axis& loopAxis) const {
    for (auto& currAxis : axis) {
        if (currAxis == loopAxis) {
            return true;
        }
    }
    return false;
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
    if (overridenInnerSizes.has_value()) {
        return overridenInnerSizes.value();
    }
    OklLoopInfo::OptSizes ret{1, 1, 1};

    if (isRegular() && children.empty()) {
        return ret;
    }

#ifdef LEGACY_INNER_SIZES_CALCULATION
    if (!children.empty()) {
        ret = children.front().getInnerSizes();
    }
#else
    size_t prevProd = 0;
    for (auto& child : children) {
        auto currSizes = child.getInnerSizes();
        auto prod = currSizes.product();
        if (prod > prevProd) {
            ret = currSizes;
        }
        prevProd = prod;
    }
#endif

    if (has(LoopType::Inner)) {
        if (type.size() == 1) {
            ret[static_cast<size_t>(axis[0])] =
                range.size == 0 ? std::nullopt : std::make_optional(range.size);
        } else if (type.size() == 2) {  // Tiled loop
            // TODO: maybe reuse attribute parser
            char* p;
            auto tileSizeLL = std::strtoll(tileSize.c_str(), &p, 10);

            // if tile size is known at compile time, then it is a size of the second loop
            if (type[1] == LoopType::Inner) {
                ret[static_cast<size_t>(axis[1])] =
                    *p ? std::nullopt : std::make_optional(static_cast<size_t>(tileSizeLL));
            }
            // if both tile size and range size are known at compile time, then ceil(range size /
            // tile size) is a size of loop
            if (type[0] == LoopType::Inner) {
                ret[static_cast<size_t>(axis[0])] = *p || range.size == 0
                                                        ? std::nullopt
                                                        : std::make_optional(static_cast<size_t>(
                                                              1 + ((range.size - 1) / tileSizeLL)));
            }
        }
    }
    return ret;
}

size_t OklLoopInfo::getHeight() {
    auto* currLoop = this;
    int h = 0;
    while (currLoop && !currLoop->children.empty()) {
        currLoop = currLoop->getFirstAttributedChild();
        if (currLoop) {
            h += currLoop->type.size();
        }
    }
    return h;
}

size_t OklLoopInfo::getHeightSameType(const LoopType& type) {
    auto* currLoop = this;
    int h = 0;
    while (currLoop && !currLoop->children.empty()) {
        currLoop = currLoop->getFirstAttributedChild();
        if (currLoop) {
            for (auto& loopType : currLoop->type) {
                if (loopType == type) {
                    ++h;
                }
            }
        }
    }
    return h;
}

bool OklLoopInfo::updateAutoWithSpecificAxis() {
    if (isTiled()) {
        auto height1 = getHeightSameType(type[0]);
        auto height2 = getHeightSameType(type[1]);
        if (type[0] == type[1]) {
            ++height1;
        }

        if (height1 > MAX_AXIS_SZ || height2 > MAX_AXIS_SZ) {
            return false;
        }

        if (axis[0] == Axis::Auto) {
            axis[0] = static_cast<Axis>(height1);
        }
        if (axis[1] == Axis::Auto) {
            axis[1] = static_cast<Axis>(height2);
        }

    } else {
        if (axis[0] != Axis::Auto) {
            return true;
        }
        if (parent == nullptr) {
            int a = 0;
        }
        auto height = getHeightSameType(type[0]);
        if (height > MAX_AXIS_SZ) {
            return false;
        }
        axis[0] = static_cast<Axis>(height);
    }
    return true;
}

bool OklLoopInfo::isLastOuter() {
    // Should have outer, but in case of @tile, other loop can be only outer or regular
    bool ok = has(LoopType::Outer) && !has(LoopType::Inner);

    auto childLoop = getFirstAttributedChild();
    if (!childLoop) {
        return false;
    }
    // Child should have inner, but in case of @tile, other loop can be only inner or regular
    ok = ok && childLoop->has(LoopType::Inner) && !childLoop->has(LoopType::Outer);

    return ok;
}

size_t OklLoopInfo::OptSizes::product() {
    return std::accumulate(begin(), end(), size_t{1}, [](const OptSize& a, const OptSize& b) {
        size_t aZ = a.value_or(size_t{1});
        size_t bZ = b.value_or(size_t{1});
        return aZ * bZ;
    });
}
bool OklLoopInfo::OptSizes::hasNullOpts() {
    return std::any_of(begin(), end(), [](const auto& val) { return val == std::nullopt; });
}

bool OklLoopInfo::OptSizes::allNullOpts() {
    return std::all_of(begin(), end(), [](const auto& val) { return val == std::nullopt; });
}

}  // namespace oklt
