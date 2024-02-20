#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>
namespace oklt {

enum class LoopOrder {
    First,
    Second,
};

enum class LoopType {
    Regular,
    Inner,
    Outer,
};

enum class Dim {
    X = 0,
    Y = 1,
    Z = 2,
};

struct AttributedLoop {
    LoopType type = LoopType::Regular;
    Dim dim = Dim::X;
};

// TODO: inner and outer can have arguments
struct TileParams {
    int tileSize;
    AttributedLoop firstLoop = AttributedLoop{};
    AttributedLoop secondLoop = AttributedLoop{};
    bool check = true;
};
}  // namespace oklt
