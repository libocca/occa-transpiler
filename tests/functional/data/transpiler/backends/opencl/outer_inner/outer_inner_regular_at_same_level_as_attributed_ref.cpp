#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_test_kernel_0();

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_test_kernel_0() {
  {
    int i = (0) + get_group_id(1);
    {
      int i2 = (0) + get_group_id(0);
      { int j = (0) + get_local_id(0); }
      for (int ii = 0; ii < 10; ++ii) {
        {
          int j = (0) + get_local_id(0);
        }
        for (int j = 0; j < 10; ++j) {
        }
      }
    }
    for (int ii = 0; ii < 10; ++ii) {
      {
        int i = (0) + get_group_id(0);
        { int j = (0) + get_local_id(0); }
      }
    }
  }
}

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_test_kernel_1();

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_test_kernel_1() {
  {
    int i = (0) + get_group_id(1);
    for (int i2 = 0; i2 < 10; ++i2) {
      {
        int i2 = (0) + get_group_id(0);
        { int j = (0) + get_local_id(0); }
      }
    }
  }
}
