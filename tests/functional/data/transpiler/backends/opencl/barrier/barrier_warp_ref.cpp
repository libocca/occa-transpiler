#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_test_kern_0();

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_test_kern_0() {
  {
    int i = (0) + get_group_id(0);
    {
      int j = (0) + get_local_id(0);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}
