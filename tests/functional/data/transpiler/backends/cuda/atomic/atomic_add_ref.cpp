#include <cuda_runtime.h>

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_add_builtin_0(const int *iVec,
                                                      int *iSum,
                                                      const float *fVec,
                                                      float *fSum) {
  atomicAdd(&(*iSum), iVec[0]);
  atomicAdd(&(*fSum), fVec[0]);
}

struct ComplexTypeF32 {
  float real;
  float imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_add_struct_0(const ComplexTypeF32 *vec,
                                                     ComplexTypeF32 *sum) {
  atomicAdd(&(sum->real), vec[0].real);
  atomicAdd(&(sum->imag), vec[0].imag);
}

template <class T> struct ComplexType {
  T real;
  T imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void
_occa_atomic_add_template_0(const ComplexType<float> *vec,
                            ComplexType<float> *sum) {
  atomicAdd(&(sum->real), vec[0].real);
  atomicAdd(&(sum->imag), vec[0].imag);
}
