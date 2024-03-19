#include <hip/hip_runtime.h>

struct S {
  int hello[12];
};

extern __constant__ int arr_0[];
extern __constant__ float arr_1[];
extern __constant__ S arr_2[];
