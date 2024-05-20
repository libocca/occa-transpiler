#pragma OPENCL EXTENSON cl_khr_fp64 : enable

struct ComplexValueFloat {
  float real;
  float imaginary;
};

__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void
_occa_function1_0(__global const int *data);

__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void
_occa_function1_0(__global const int *data) {
  {
    int i = (0) + get_group_id(0);
    __local ComplexValueFloat arr2[8][32];
    __local ComplexValueFloat arr1[32];
    { int j = (0) + get_local_id(0); }
  }
}
