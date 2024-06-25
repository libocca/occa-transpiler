#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_exch_builtin_0(__global const int *iVec, __global int *iSum,
                            __global const float *fVec, __global float *fSum);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_exch_builtin_0(__global const int *iVec, __global int *iSum,
                            __global const float *fVec, __global float *fSum) {
  {
    int i = (0) + get_group_id(0);
    {
      int j = (0) + get_local_id(0);
      *iSum = iVec[0];
      *fSum = fVec[0];
    }
  }
}

struct ComplexTypeF32 {
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

template <class T> struct ComplexType {
  T real;
  T imag;
};

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_exch_template_0(__global const ComplexType<float> *vec,
                             __global ComplexType<float> *result);

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void
_occa_atomic_exch_template_0(__global const ComplexType<float> *vec,
                             __global ComplexType<float> *result) {
  {
    int i = (0) + get_group_id(0);
    {
      int j = (0) + get_local_id(0);
      *result = vec[0];
    }
  }
}
