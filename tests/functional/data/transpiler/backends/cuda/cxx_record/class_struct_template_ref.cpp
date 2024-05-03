#include <cuda_runtime.h>

template <typename T> struct ComplexType {
  T v1;
  T v2;
  __device__ T calc();

  __device__ ComplexType(T in) : v1(in), v2(in) {}

  template <typename U> __device__ U calc(T in);
};

struct ComplexTypeFloat {
  float v1;
  float v2;
  __device__ float calc();
  template <typename T> __device__ ComplexTypeFloat(T in);
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
