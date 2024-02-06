#pragma once

#include <oklt/core/config.h>

namespace oklt {

enum struct SEMANTIC_CATEGORY {
  HOST_KERNEL_CATEGORY,
  DEVICE_KERNEL_CATEGORY,
};

SEMANTIC_CATEGORY fromBackendType(TRANSPILER_TYPE backend);

}
