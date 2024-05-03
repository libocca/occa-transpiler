#include <cuda_runtime.h>

namespace A {
template <class T> struct Complex {
  T real;
  T imaginary;
};

namespace B {
struct Configs {
  unsigned int size1;
  unsigned long size2;
};

namespace C {
typedef int SIZE_TYPE;
typedef SIZE_TYPE SIZES;
} // namespace C
} // namespace B
} // namespace A


extern "C" __global__ __launch_bounds__(1) void _occa_function1_0(
    const A::Complex<float> *__restrict__ vectorData, unsigned int vectorSize,
    const A::Complex<float> **__restrict__ matricesData,
    const A::B::Configs *__restrict__ matricesSizes) {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}


extern "C" __global__ __launch_bounds__(1) void _occa_function2_0(
    const A::Complex<float> *__restrict__ vectorData,
    const A::B::Configs *__restrict__ configs,
    A::B::C::SIZES *__restrict__ vectorSize) {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
