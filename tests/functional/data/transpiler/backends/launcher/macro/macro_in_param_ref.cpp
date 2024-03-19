#include <occa/core/kernel.hpp>

extern "C" void mykern(occa::modeKernel_t** deviceKernels, int& aaa, int& bbb) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int i = 0;
        outer[0] = (10) - (0);
        int j = 0;
        inner[0] = (10) - (0);
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(aaa, bbb);
    };
}
