#include <cuda_runtime.h>
// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_sub_builtin_0(const int *iVec,
                                                      int *iSum,
                                                      const float *fVec,
                                                      float *fSum) {
  atomicSub(&(*iSum), iVec[0]);
  atomicSub(&(*fSum), fVec[0]);
}

struct ComplexTypeF32 {
  float real;
  float imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void _occa_atomic_sub_struct_0(const ComplexTypeF32 *vec,
                                                     ComplexTypeF32 *sum) {
  atomicSub(&(sum->real), vec[0].real);
  atomicSub(&(sum->imag), vec[0].imag);
}

template <class T> struct ComplexType {
  T real;
  T imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void
_occa_atomic_sub_template_0(const ComplexType<float> *vec,
                            ComplexType<float> *sum) {
  atomicSub(&(sum->real), vec[0].real);
  atomicSub(&(sum->imag), vec[0].imag);
}
