#include <occa/core/kernel.hpp>

int* __restrict__ okl(int a, int b) {
    static int c = a + b;
    return &c;
}

extern "C" void add_kernel(occa::modeKernel_t** deviceKernels,
                           const int& n,
                           occa::modeMemory_t* a,
                           occa::modeMemory_t* b,
                           occa::modeMemory_t* c) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[0] = ((n) - (0) + 16 - 1) / 16;
        int i = _occa_tiled_i;
        inner[0] = (_occa_tiled_i + 16) - _occa_tiled_i;
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(n, a, b, c);
    };
}
