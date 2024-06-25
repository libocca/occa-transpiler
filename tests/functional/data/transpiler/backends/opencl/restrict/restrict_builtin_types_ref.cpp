#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_function1_0(__global const int *restrict i32Data,
                  __global float *restrict fp32Data,
                  __global const double *restrict fp64Data);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_function1_0(__global const int *restrict i32Data,
                  __global float *restrict fp32Data,
                  __global const double *restrict fp64Data) {
  {
    int i = (0) + get_group_id(0);
    {
      int j = (0) + get_local_id(0);
      float *b = &fp32Data[0];
    }
  }
}
