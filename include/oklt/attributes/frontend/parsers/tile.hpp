#include <clang/AST/Attr.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt {
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

struct Loop {
    LoopType type = LoopType::Regular;
    Dim dim = Dim::X;
};

// TODO: inner and outer can have arguments
struct TileParams {
    int tileSize;
    Loop firstLoop = Loop{};
    Loop secondLoop = Loop{};
    bool check = true;
};

bool parseTileAttribute(const clang::Attr* a, SessionStage& s);
}  // namespace oklt
