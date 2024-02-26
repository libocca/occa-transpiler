#pragma once

#include <string>
#include "attributes/frontend/params/loop.h"

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