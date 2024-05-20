#pragma OPENCL EXTENSON cl_khr_fp64 : enable

// const array
__constant int arr_const0[12] = {0};
__constant int arr_const1[12] = {0};
// Stupid formatting
__constant int arr_const2[12] = {0};
// Deduced size
__constant float arr_const3[] = {1., 2., 3., 4., 5., 6.};
// Multidimensional
__constant float arr_const4[][2] = {{1., 2.}, {3., 4.}, {5., 6.}};
__constant float arr_const5[][3][2] = {{{1., 2.}, {3., 4.}, {5., 6.}},
                                       {{1., 2.}, {3., 4.}, {5., 6.}}};
// At least one @kern function is requried
__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void _occa_kern_0();

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void _occa_kern_0() {
  {
    int i = (0) + get_group_id(0);
    { int j = (0) + get_local_id(0); }
  }
}
