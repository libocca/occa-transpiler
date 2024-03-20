#include <cuda_runtime.h>

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ __launch_bounds__(1) void _occa_atomic_exch_builtin_0(
    const int *iVec, int *iSum, const float *fVec, float *fSum) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicExch(&(*iSum), iVec[0]);
      atomicExch(&(*fSum), fVec[0]);
    }
  }
}

struct ComplexTypeF32 {
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

template <class T> struct ComplexType {
  T real;
  T imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ __launch_bounds__(1) void _occa_atomic_exch_template_0(
    const ComplexType<float> *vec, ComplexType<float> *result) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicExch(&(*result), vec[0]);
    }
  }
}
