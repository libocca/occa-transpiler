#include <occa/core/kernel.hpp>

extern "C" void test0(occa::modeKernel_t** deviceKernels,
                      const int& entries,
                      occa::modeMemory_t* a,
                      occa::modeMemory_t* b,
                      occa::modeMemory_t* ab) {
    int before0 = 0;
    {
        occa::dim outer, inner;
        outer.dims = 3;
        inner.dims = 3;
        int x = 0;
        outer[2] = (10) - (0);
        int y = 0;
        outer[1] = (20) - (0);
        int z = 0;
        outer[0] = (30) - (0);
        int n = 0;
        inner[2] = (2) - (0);
        int m = 0;
        inner[1] = (3) - (0);
        int k = 0;
        inner[0] = (5) - (0);
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
    int before00 = 1 + before0;
    {
        occa::dim outer, inner;
        outer.dims = 3;
        inner.dims = 3;
        int x = 0;
        outer[2] = (10) - (0);
        int y = 0;
        outer[1] = (20) - (0);
        int z = 0;
        outer[0] = (30) - (0);
        int n = 0;
        inner[2] = (2) - (0);
        int m = 0;
        inner[1] = (3) - (0);
        int k = 0;
        inner[0] = (5) - (0);
        occa::kernel kernel(deviceKernels[1]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}
