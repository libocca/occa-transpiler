#include <occa/core/kernel.hpp>

const int offset = 1;

// template<typename T>
float add(float a, float b) { return a + b + offset; }

// Outer -> inner ==> regular -> regular
extern "C" void addVectors0(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[0] = ((entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int i = _occa_tiled_i;
        inner[0] = ((_occa_tiled_i + 4) - _occa_tiled_i + (1) - 1) / (1);
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> inner ==> inner -> regular
extern "C" void addVectors2(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 2;
        int _occa_tiled_i = 0;
        outer[0] = ((entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int i = _occa_tiled_i;
        inner[1] = ((_occa_tiled_i + 4) - _occa_tiled_i + (1) - 1) / (1);
        int _occa_tiled_j = 0;
        inner[0] = ((entries) - (0) + 4 - 1) / 4;
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> inner ==> inner -> inner
extern "C" void addVectors3(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 3;
        int _occa_tiled_i = 0;
        outer[0] = ((entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int i = _occa_tiled_i;
        inner[2] = ((_occa_tiled_i + 4) - _occa_tiled_i + (1) - 1) / (1);
        int _occa_tiled_j = 0;
        inner[1] = ((entries) - (0) + 4 - 1) / 4;
        int j = _occa_tiled_j;
        inner[0] = (_occa_tiled_j + 4) - _occa_tiled_j;
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> outer ==> inner -> regular
extern "C" void addVectors4(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
    {
        occa::dim outer, inner;
        outer.dims = 2;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[1] = ((entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int i = _occa_tiled_i;
        outer[0] = ((_occa_tiled_i + 4) - _occa_tiled_i + (1) - 1) / (1);
        int _occa_tiled_j = 0;
        inner[0] = ((entries) - (0) + 4 - 1) / 4;
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> outer ==> inner -> inner
extern "C" void addVectors5(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
    {
        occa::dim outer, inner;
        outer.dims = 2;
        inner.dims = 2;
        int _occa_tiled_i = 0;
        outer[1] = ((entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int i = _occa_tiled_i;
        outer[0] = ((_occa_tiled_i + 4) - _occa_tiled_i + (1) - 1) / (1);
        int _occa_tiled_j = 0;
        inner[1] = ((entries) - (0) + 4 - 1) / 4;
        int j = _occa_tiled_j;
        inner[0] = (_occa_tiled_j + 4) - _occa_tiled_j;
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> outer ==> outer -> inner
extern "C" void addVectors6(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
    {
        occa::dim outer, inner;
        outer.dims = 3;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[2] = ((entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int i = _occa_tiled_i;
        outer[1] = ((_occa_tiled_i + 4) - _occa_tiled_i + (1) - 1) / (1);
        int _occa_tiled_j = 0;
        outer[0] = ((entries) - (0) + 4 - 1) / 4;
        int j = _occa_tiled_j;
        inner[0] = (_occa_tiled_j + 4) - _occa_tiled_j;
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}
