#pragma OPENCL EXTENSON cl_khr_fp64 : enable

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

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_function1_0(__global const A::Complex<float> *restrict vectorData,
                  unsigned int vectorSize,
                  __global const A::Complex<float> **restrict matricesData,
                  __global const A::B::Configs *restrict matricesSizes);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_function1_0(__global const A::Complex<float> *restrict vectorData,
                  unsigned int vectorSize,
                  __global const A::Complex<float> **restrict matricesData,
                  __global const A::B::Configs *restrict matricesSizes) {
  {
    int i = (0) + get_group_id(0);
    { int j = (0) + get_local_id(0); }
  }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_function2_0(__global const A::Complex<float> *restrict vectorData,
                  __global const A::B::Configs *restrict configs,
                  __global A::B::C::SIZES *restrict vectorSize);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_function2_0(__global const A::Complex<float> *restrict vectorData,
                  __global const A::B::Configs *restrict configs,
                  __global A::B::C::SIZES *restrict vectorSize) {
  {
    int i = (0) + get_group_id(0);
    { int j = (0) + get_local_id(0); }
  }
}
