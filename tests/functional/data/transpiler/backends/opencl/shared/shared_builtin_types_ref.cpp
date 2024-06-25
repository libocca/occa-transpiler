#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void
_occa_function1_0(__global const int *data);

__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void
_occa_function1_0(__global const int *data) {
  {
    int i = (0) + get_group_id(0);
    __local int arr1[32];
    __local float arr2[8][32];
    __local double arr3[4 + 4];
    { int j = (0) + get_local_id(0); }
  }
}

// syncronization between @inner loops:
__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_function2_0();

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_function2_0() {
  {
    int i = (0) + get_group_id(0);
    __local int shm[10];
    {
      int j = (0) + get_local_id(0);
      shm[i] = j;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // sync should be here
    {
      int j = (0) + get_local_id(0);
      shm[i] = j;
    }
    // sync should not be here
  }
}

// Even if loop is last, if it is inside regular loop, syncronization is
// inserted
__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_function3_0();

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_function3_0() {
  {
    int i = (0) + get_group_id(0);
    __local int shm[10];
    for (int q = 0; q < 5; ++q) {
      {
        int j = (0) + get_local_id(0);
        shm[i] = j;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      // sync should be here
    }
  }
}
