#include <cuda_runtime.h>

template <class T> struct Complex {
  T real;
  T imaginary;
};

struct Configs {
  unsigned int size1;
  unsigned long size2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" __global__ void
_occa_function1_0(const Complex<float> *__restrict__ vectorData,
                  unsigned int vectorSize,
                  const Complex<float> **__restrict__ matricesData,
                  const Configs *__restrict__ matricesSizes) {}
