#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(4, 3, 1))) void
_occa_test_kern_0();

__kernel __attribute__((reqd_work_group_size(4, 3, 1))) void
_occa_test_kern_0() {
  {
    int _occa_tiled_i = (0) + ((4) * get_group_id(0));
    for (int i = _occa_tiled_i; i < (_occa_tiled_i + (4)); ++i) {
      if (i < 10) {
        __local int shm[10];
        {
          int _occa_tiled_j = (0) + ((4) * get_local_id(1));
          {
            int j = _occa_tiled_j + get_local_id(0);
            if (j < 10) {
              shm[j] = j;
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
  }
}
