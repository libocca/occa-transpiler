#include "attributes/utils/common.h"

namespace oklt {
bool isLastOuter(OklLoopInfo* loop) {
    if (!loop) {
        return false;
    }
    // Should have outer, but in case of @tile, other loop can be only outer or regular
    bool ok = loop->has(LoopType::Outer) && !loop->has(LoopType::Inner);

    auto childLoop = loop->getFirstAttributedChild();
    if (!childLoop) {
        return false;
    }
    // Child should have inner, but in case of @tile, other loop can be only inner or regular
    ok = ok && childLoop->has(LoopType::Inner) && !childLoop->has(LoopType::Outer);

    return ok;
}
}  // namespace oklt
