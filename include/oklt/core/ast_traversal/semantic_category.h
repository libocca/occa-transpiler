#pragma once

#include <oklt/core/target_backends.h>

namespace oklt {

enum struct SemanticCategory {
  HOST_KERNEL_CATEGORY,
  DEVICE_KERNEL_CATEGORY,
};

SemanticCategory fromBackendType(TargetBackend backend);

}
