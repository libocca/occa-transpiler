#pragma OPENCL EXTENSON cl_khr_fp64 : enable

template <class T> struct ComplexType {
  T real;
  T imaginary;
};

__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void
_occa_function1_0(__global const int *data);

__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void
_occa_function1_0(__global const int *data) {
  {
    int i = (0) + get_group_id(0);
    __local ComplexType<int> arr1[32];
    __local ComplexType<float> arr2[8][32];
    { int j = (0) + get_local_id(0); }
  }
}
