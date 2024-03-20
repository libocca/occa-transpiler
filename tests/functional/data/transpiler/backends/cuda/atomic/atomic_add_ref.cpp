#include <cuda_runtime.h>


extern "C" __global__ __launch_bounds__(1) void _occa_atomic_add_builtin_0(
    const int *iVec, int *iSum, const float *fVec, float *fSum) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicAdd(&(*iSum), iVec[0]);
      atomicAdd(&(*fSum), fVec[0]);
    }
  }
}

struct ComplexTypeF32 {
  float real;
  float imag;
};


extern "C" __global__
__launch_bounds__(1) void _occa_atomic_add_struct_0(const ComplexTypeF32 *vec,
                                                    ComplexTypeF32 *sum) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicAdd(&(sum->real), vec[0].real);
      atomicAdd(&(sum->imag), vec[0].imag);
    }
  }
}

template <class T> struct ComplexType {
  T real;
  T imag;
};


extern "C" __global__ __launch_bounds__(1) void _occa_atomic_add_template_0(
    const ComplexType<float> *vec, ComplexType<float> *sum) {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      atomicAdd(&(sum->real), vec[0].real);
      atomicAdd(&(sum->imag), vec[0].imag);
    }
  }
}
