#include "attributes/frontend/params/loop.h"

namespace oklt {
struct InnerParams {
    ParsedLoopParams attribute_params;
    struct {
        bool synchronize_loop = false;
    } sema_params;
};

}  // namespace  oklt
