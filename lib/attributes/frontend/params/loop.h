#pragma once
#include <oklt/core/kernel_metadata.h>

#include <clang/AST/Attr.h>

namespace oklt {

enum class Axis {
    X = 0,
    Y = 1,
    Z = 2,
    Auto = 3,
};

constexpr auto MAX_AXIS = Axis::Z;
constexpr auto MAX_AXIS_SZ = static_cast<size_t>(Axis::Z);
constexpr auto N_AXIS = MAX_AXIS_SZ + 1;

struct AttributedLoop {
    LoopType type = LoopType::Regular;
    Axis axis = Axis::Auto;
};

struct AttributedLoopInnerSize {
    std::array<int, N_AXIS> size = {-1, -1, -1};
};

}  // namespace oklt
