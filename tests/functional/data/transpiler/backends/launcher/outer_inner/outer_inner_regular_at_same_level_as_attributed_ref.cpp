#include <occa/core/kernel.hpp>

extern "C" void test_kernel(occa::modeKernel_t **deviceKernels) {
  {
    occa::dim outer, inner;
    outer.dims = 2;
    inner.dims = 1;
    int i = 0;
    outer[1] = (10) - (0);
    int i2 = 0;
    outer[0] = (10) - (0);
    int j = 0;
    inner[0] = (10) - (0);
    occa::kernel kernel(deviceKernels[0]);
    kernel.setRunDims(outer, inner);
    kernel();
  };
  for (int ii = 0; ii < 10; ++ii) {
    {
      occa::dim outer, inner;
      outer.dims = 2;
      inner.dims = 1;
      int i = 0;
      outer[1] = (10) - (0);
      int i2 = 0;
      outer[0] = (10) - (0);
      int j = 0;
      inner[0] = (10) - (0);
      occa::kernel kernel(deviceKernels[1]);
      kernel.setRunDims(outer, inner);
      kernel();
    };
  }
}
