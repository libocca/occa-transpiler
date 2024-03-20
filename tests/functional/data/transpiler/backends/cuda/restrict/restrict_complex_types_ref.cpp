#include <cuda_runtime.h>

template <class T> struct Complex {
  T real;
  T imaginary;
};

struct Configs {
  unsigned int size1;
  unsigned long size2;
};

struct Data {
  float *__restrict__ x;
  float *__restrict__ y;
  unsigned long size;
};


extern "C" __global__ __launch_bounds__(1) void _occa_function1_0(
    const Complex<float> *__restrict__ vectorData, unsigned int vectorSize,
    const Complex<float> **__restrict__ matricesData,
    const Configs *__restrict__ matricesSizes) {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
