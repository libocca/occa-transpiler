#pragma once

#include "loop.h"
#include <string>

namespace oklt {
enum class LoopOrder {
    First,
    Second,
};

struct TileParams {
    std::string tileSize;
    AttributedLoop firstLoop = AttributedLoop{};
    AttributedLoop secondLoop = AttributedLoop{};
    bool check = true;
};
}  // namespace oklt
