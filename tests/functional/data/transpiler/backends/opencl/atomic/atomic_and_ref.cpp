#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_and_builtin_0(__global const unsigned int *masks,
                           __global unsigned int *mask);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_and_builtin_0(__global const unsigned int *masks,
                           __global unsigned int *mask) {
  {
    int i = (0) + get_group_id(0);
    {
      int j = (0) + get_local_id(0);
      *mask &= masks[0];
    }
  }
}

struct ComplexMaskType {
  unsigned int mask1;
  unsigned int mask2;
};

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_and_struct_0(__global const ComplexMaskType *masks,
                          __global ComplexMaskType *mask);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_and_struct_0(__global const ComplexMaskType *masks,
                          __global ComplexMaskType *mask) {
  {
    int i = (0) + get_group_id(0);
    {
      int j = (0) + get_local_id(0);
      mask->mask1 &= masks[0].mask1;
      mask->mask2 &= masks[0].mask2;
    }
  }
}
