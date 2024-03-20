#include <cuda_runtime.h>

extern "C" __global__ void _occa_hello_kern_0() {
  {
    int i = (0) + blockIdx.x;
    {
      int j = (0) + threadIdx.x;
      float var = 10.0;
      float res = __exp10f(var);
      auto ok = std::isnan(var);
    }
  }
}
