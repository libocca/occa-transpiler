#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_dec_builtin_0(__global unsigned int *value);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_dec_builtin_0(__global unsigned int *value) {
  {
    int i = (0) + get_group_id(0);
    {
      int j = (0) + get_local_id(0);
      --(*value);
    }
  }
}

struct ComplexMaskType {
  unsigned int val1;
  int val2;
};

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_dec_struct_0(__global ComplexMaskType *value);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_dec_struct_0(__global ComplexMaskType *value) {
  {
    int i = (0) + get_group_id(0);
    {
      int j = (0) + get_local_id(0);
      --value->val1;
      value->val2--;
    }
  }
}
