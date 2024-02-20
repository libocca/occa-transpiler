#pragma once

#include "loop.h"

namespace oklt {
enum class LoopOrder {
    First,
    Second,
};

struct TileParams {
    int tileSize;
    AttributedLoop firstLoop = AttributedLoop{};
    AttributedLoop secondLoop = AttributedLoop{};
    bool check = true;
};
}  // namespace oklt
