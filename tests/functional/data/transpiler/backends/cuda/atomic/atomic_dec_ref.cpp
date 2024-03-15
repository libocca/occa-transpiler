#include <cuda_runtime.h>

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_dec_builtin_0(unsigned int *value) {
  atomicDec(&((*value)), 1);
  // @atomic (*value)--; normalizer issue
}

struct ComplexMaskType {
  unsigned int val1;
  int val2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_dec_struct_0(ComplexMaskType *value) {
  atomicDec(&(value->val1), 1);
  atomicDec(&(value->val2), 1);
}
