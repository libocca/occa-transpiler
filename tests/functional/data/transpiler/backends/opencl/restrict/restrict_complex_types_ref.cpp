#pragma OPENCL EXTENSON cl_khr_fp64 : enable

template <class T> struct Complex {
  T real;
  T imaginary;
};

struct Configs {
  unsigned int size1;
  unsigned long size2;
};

struct Data {
  float *x;
  float *y;
  unsigned long size;
};

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_function1_0(__global const Complex<float> *restrict vectorData,
                  unsigned int vectorSize,
                  __global const Complex<float> **restrict matricesData,
                  __global const Configs *restrict matricesSizes);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_function1_0(__global const Complex<float> *restrict vectorData,
                  unsigned int vectorSize,
                  __global const Complex<float> **restrict matricesData,
                  __global const Configs *restrict matricesSizes) {
  {
    int i = (0) + get_group_id(0);
    { int j = (0) + get_local_id(0); }
  }
}
