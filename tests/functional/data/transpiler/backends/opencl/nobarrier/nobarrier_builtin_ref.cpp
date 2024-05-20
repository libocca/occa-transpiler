#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_hello_kern_0();

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void
_occa_hello_kern_0() {
  {
    int i = (0) + get_group_id(0);
    __local int shm[10];
    {
      int j = (0) + get_local_id(0);
      shm[j] = j;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
      int j = (0) + get_local_id(0);
      shm[j] = j;
    }
    {
      int j = (0) + get_local_id(0);
      shm[j] = j;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
      int j = (0) + get_local_id(0);
      shm[j] = j;
    }
  }
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void
_occa_priority_issue_0();

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void
_occa_priority_issue_0() {
  {
    int i = (0) + get_group_id(0);
    __local float shm[32];
    {
      int j = (0) + get_local_id(0);
      shm[i] = i;
    }
    {
      int j = (0) + get_local_id(0);
      shm[i * j] += 32;
    }
  }
}
