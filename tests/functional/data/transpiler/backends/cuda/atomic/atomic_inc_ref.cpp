#include <cuda_runtime.h>


extern "C" __global__
__launch_bounds__(1) void _occa_atomic_inc_builtin_0(unsigned int *value) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicInc(&((*value)), 1);
    }
  }
}

struct ComplexMaskType {
  unsigned int val1;
  int val2;
};


extern "C" __global__
__launch_bounds__(1) void _occa_atomic_inc_struct_0(ComplexMaskType *value) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicInc(&(value->val1), 1);
      atomicInc(&(value->val2), 1);
    }
  }
}
