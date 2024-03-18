#include <cuda_runtime.h>

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_and_builtin_0(const unsigned int *masks,
                                                      unsigned int *mask) {
  atomicOr(&(*mask), masks[0]);
}

struct ComplexMaskType {
  unsigned int mask1;
  unsigned int mask2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void
_occa_atomic_and_struct_0(const ComplexMaskType *masks, ComplexMaskType *mask) {
  atomicOr(&(mask->mask1), masks[0].mask1);
  atomicOr(&(mask->mask2), masks[0].mask2);
}
