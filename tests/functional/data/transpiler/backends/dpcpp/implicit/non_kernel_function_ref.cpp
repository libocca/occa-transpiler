#include <CL/sycl.hpp>
using namespace sycl;

SYCL_EXTERNAL static float add1(const float *a, int i, const float *b, int j) {
  return a[i] + b[i];
}

SYCL_EXTERNAL float add2(const float *a, int i, const float *b, int j) {
  return a[i] + b[i];
}
