#include <cuda_runtime.h>
template <int aa, int bb> class HelloClass;

template <int bb> class HelloClass<0, bb> {
public:
  __device__ static inline void myfn() {}
};

template <int bb> class HelloClassFull {
public:
  __device__ inline void myfn() {}
};

template <> class HelloClassFull<0> {
public:
  __device__ inline void myfn() {}
};

extern "C" __global__
__launch_bounds__(16) void _occa_reductionWithSharedMemory_0(const int entries,
                                                             const float *vec) {
  {
    int _occa_tiled_i = (0) + ((16) * blockIdx.x);
    {
      int i = _occa_tiled_i + threadIdx.x;
      if (i < entries) {
        auto tmp = vec[i];
      }
    }
  }
}
