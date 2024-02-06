#include <oklt/core/ast_traversal/semantic_category.h>

namespace oklt {

SEMANTIC_CATEGORY fromBackendType(TRANSPILER_TYPE backend) {
  switch (backend) {
    case TRANSPILER_TYPE::CUDA:
      return SEMANTIC_CATEGORY::DEVICE_KERNEL_CATEGORY;
    case TRANSPILER_TYPE::OPENMP:
      return SEMANTIC_CATEGORY::HOST_KERNEL_CATEGORY;
  }
  return SEMANTIC_CATEGORY::HOST_KERNEL_CATEGORY;
}
}
