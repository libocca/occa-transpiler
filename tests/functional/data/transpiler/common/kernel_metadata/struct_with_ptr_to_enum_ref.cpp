#include <cuda_runtime.h>

typedef enum { El1, El2, El3 } MyEnum;

typedef struct {
  MyEnum *my_elems;
} MyStruct;

extern "C" __global__ __launch_bounds__(3) void _occa_kern_0(MyStruct s) {
  {
    int i = (0) + blockIdx.x;
    { int j = (0) + threadIdx.x; }
  }
}
