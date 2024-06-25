#pragma OPENCL EXTENSON cl_khr_fp64 : enable

// pointer to const
__constant int *ptr_const0 = 0;
__constant int *ptr_const1 = 0;
// const pointer to const
__constant int *const ptr_const2 = 0;
__constant int *const ptr_const3 = 0;
// const pointer to non const
int *const ptr_const4 = 0;
// Stupid formatting
__constant int *ptr_const5 = 0;
// At least one @kern function is requried
__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void _occa_kern_0();

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void _occa_kern_0() {
  {
    int i = (0) + get_group_id(0);
    { int j = (0) + get_local_id(0); }
  }
}
