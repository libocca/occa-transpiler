#pragma OPENCL EXTENSON cl_khr_fp64 : enable

typedef float ex_float32_t;
__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void
_occa_test_kernel_0();

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void
_occa_test_kernel_0() {
  {
    int i = (0) + get_group_id(0);
    ex_float32_t d[32];
    {
      int j = (0) + get_local_id(0);
      d[j] = i - j;
    }
  }
}
