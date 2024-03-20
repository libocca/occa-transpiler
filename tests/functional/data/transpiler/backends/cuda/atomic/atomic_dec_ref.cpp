#include <cuda_runtime.h>

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__
__launch_bounds__(1) void _occa_atomic_dec_builtin_0(unsigned int *value) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicDec(&((*value)), 1);
    }
  }
}

struct ComplexMaskType {
  unsigned int val1;
  int val2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__
__launch_bounds__(1) void _occa_atomic_dec_struct_0(ComplexMaskType *value) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicDec(&(value->val1), 1);
      atomicDec(&(value->val2), 1);
    }
  }
}
