#include <cuda_runtime.h>

struct ComplexTypeF32 {
  __device__ ComplexTypeF32 &operator=(const ComplexTypeF32 &) = default;
  float real;
  float imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__
__launch_bounds__(1) void _occa_atomic_exch_struct_0(const ComplexTypeF32 *vec,
                                                     ComplexTypeF32 *result) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicExch(&(*result), vec[0]);
    }
  }
}
