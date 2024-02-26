#include <cuda_runtime.h>
// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_inc_builtin_0(unsigned int *value) {
  atomicInc(&((*value)), 1);
  // @atomic (*value)++; normalizer issue
}

struct ComplexMaskType {
  unsigned int val1;
  int val2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_inc_struct_0(ComplexMaskType *value) {
  atomicInc(&(value->val1), 1);
  atomicInc(&(value->val2), 1);
}
