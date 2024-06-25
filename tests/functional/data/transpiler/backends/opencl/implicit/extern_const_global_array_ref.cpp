#pragma OPENCL EXTENSON cl_khr_fp64 : enable

struct S {
  int hello[12];
};

extern __constant int arr_0[];
extern __constant float arr_1[];
extern __constant S arr_2[];
// At least one @kern function is requried
__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void _occa_kern_0();

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void _occa_kern_0() {
  {
    int i = (0) + get_group_id(0);
    { int j = (0) + get_local_id(0); }
  }
}
