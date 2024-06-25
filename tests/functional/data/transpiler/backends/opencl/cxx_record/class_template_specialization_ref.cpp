#pragma OPENCL EXTENSON cl_khr_fp64 : enable

template <int aa, int bb> class HelloClass;

template <int bb> class HelloClass<0, bb> {
public:
  static inline void myfn() {}
};

template <int bb> class HelloClassFull {
public:
  inline void myfn() {}
};

template <> class HelloClassFull<0> {
public:
  inline void myfn() {}
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
