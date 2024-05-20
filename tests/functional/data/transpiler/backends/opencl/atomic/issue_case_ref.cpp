#pragma OPENCL EXTENSON cl_khr_fp64 : enable

struct ComplexTypeF32 {
  ComplexTypeF32 &operator=(const ComplexTypeF32 &) = default;
  float real;
  float imag;
};

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_exch_struct_0(__global const ComplexTypeF32 *vec,
                           __global ComplexTypeF32 *result);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_exch_struct_0(__global const ComplexTypeF32 *vec,
                           __global ComplexTypeF32 *result) {
  {
    int i = (0) + get_group_id(0);
    {
      int j = (0) + get_local_id(0);
      *result = vec[0];
    }
  }
}
