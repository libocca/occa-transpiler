#pragma OPENCL EXTENSON cl_khr_fp64 : enable

template <typename T> struct ComplexType {
  T v1;
  T v2;
  T calc();

  ComplexType(T in) : v1(in), v2(in) {}

  template <typename U> U calc(T in);
};

struct ComplexTypeFloat {
  float v1;
  float v2;
  float calc();
  template <typename T> ComplexTypeFloat(T in);
};

__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void
_occa_reductionWithSharedMemory_0(const int entries, __global const float *vec);

__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void
_occa_reductionWithSharedMemory_0(const int entries,
                                  __global const float *vec) {
  {
    int _occa_tiled_i = (0) + ((16) * get_group_id(0));
    {
      int i = _occa_tiled_i + get_local_id(0);
      if (i < entries) {
        auto tmp = vec[i];
      }
    }
  }
}
