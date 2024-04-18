#include <hip/hip_runtime.h>
constexpr float f = 13;

class HelloClass {
public:
  static constexpr int a = 2 + 2;
};

extern "C" __global__ __launch_bounds__(10) void _occa_hello_kern_0() {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
