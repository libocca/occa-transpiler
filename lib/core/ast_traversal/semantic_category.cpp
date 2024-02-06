#include <oklt/core/ast_traversal/semantic_category.h>

namespace oklt {

SemanticCategory fromBackendType(TargetBackend backend) {
  switch (backend) {
    case TargetBackend::CUDA:
      return SemanticCategory::DEVICE_KERNEL_CATEGORY;
    case TargetBackend::OPENMP:
      return SemanticCategory::HOST_KERNEL_CATEGORY;
  }
  return SemanticCategory::HOST_KERNEL_CATEGORY;
}
}
