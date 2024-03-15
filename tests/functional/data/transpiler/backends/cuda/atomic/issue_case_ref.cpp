#include <cuda_runtime.h>

struct ComplexTypeF32 {
  ComplexTypeF32 &operator=(const ComplexTypeF32 &) = default;
  float real;
  float imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_exch_struct_0(const ComplexTypeF32 *vec,
                                                      ComplexTypeF32 *result) {
  atomicExch(&(*result), vec[0]);
}
