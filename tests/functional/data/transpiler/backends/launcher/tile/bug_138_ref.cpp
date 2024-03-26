#include <occa/core/kernel.hpp>

extern "C" void hello_kern(occa::modeKernel_t** deviceKernels) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[0] = ((1) - (0) + 1 - 1) / 1;
        int i = _occa_tiled_i;
        inner[0] = (_occa_tiled_i + 1) - _occa_tiled_i;
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel();
    };
}
