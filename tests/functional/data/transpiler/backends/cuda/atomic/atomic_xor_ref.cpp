#include <cuda_runtime.h>

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__
__launch_bounds__(1) void _occa_atomic_and_builtin_0(const unsigned int *masks,
                                                     unsigned int *mask) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicXor(&(*mask), masks[0]);
    }
  }
}

struct ComplexMaskType {
  unsigned int mask1;
  unsigned int mask2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ __launch_bounds__(1) void _occa_atomic_and_struct_0(
    const ComplexMaskType *masks, ComplexMaskType *mask) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicXor(&(mask->mask1), masks[0].mask1);
      atomicXor(&(mask->mask2), masks[0].mask2);
    }
  }
}
