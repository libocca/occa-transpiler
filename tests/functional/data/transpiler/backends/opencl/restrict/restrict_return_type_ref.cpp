#pragma OPENCL EXTENSON cl_khr_fp64 : enable

float *myfn(float *a);

float *myfn(float *a) { return a + 1; }

float *myfn2(float *a);

float *myfn2(float *a) { return a + 1; }

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void _occa_hello_0();

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void _occa_hello_0() {
  {
    int i = (0) + get_group_id(0);
    { int j = (0) + get_local_id(0); }
  }
}
