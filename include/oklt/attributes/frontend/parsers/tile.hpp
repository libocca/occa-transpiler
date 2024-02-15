#include <clang/AST/Attr.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt {
enum class LoopType {
    Regular,
    Inner,
    Outer,
};

// TODO: inner and outer can have arguments
struct TileParams {
    int tileSize;
    LoopType firstLoopType = LoopType::Regular;
    LoopType secondLoopType = LoopType::Regular;
    bool check = false;
};

bool parseTileAttribute(const clang::Attr* a, SessionStage& s);
}  // namespace oklt