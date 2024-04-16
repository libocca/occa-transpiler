#include <occa/core/kernel.hpp>

const int offset = 1;

// template<typename T>
// Outer -> inner
extern "C" void addVectors0(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
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

// Outer -> inner non 1 increment
extern "C" void addVectors1(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[0] = ((entries) - (0) + (4 * (2)) - 1) / (4 * (2));
        int i = _occa_tiled_i;
        inner[0] = ((_occa_tiled_i + 4) - _occa_tiled_i + (2) - 1) / (2);
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> inner unary post add
extern "C" void addVectors2(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[0] = ((entries) - (0) + 4 - 1) / 4;
        int i = _occa_tiled_i;
        inner[0] = (_occa_tiled_i + 4) - _occa_tiled_i;
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> inner unary pre add
extern "C" void addVectors3(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[0] = ((entries) - (0) + 4 - 1) / 4;
        int i = _occa_tiled_i;
        inner[0] = (_occa_tiled_i + 4) - _occa_tiled_i;
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> inner, check=True
extern "C" void addVectors4(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
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

// Outer -> inner, complex range
extern "C" void addVectors5(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int _occa_tiled_i = (entries - 12 + 4);
        outer[0] = (((entries + 16)) - ((entries - 12 + 4)) + (4 * ((entries / 16 + 1))) - 1) /
                   (4 * ((entries / 16 + 1)));
        int i = _occa_tiled_i;
        inner[0] =
            ((_occa_tiled_i + 4) - _occa_tiled_i + ((entries / 16 + 1)) - 1) / ((entries / 16 + 1));
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> inner, set dimension
extern "C" void addVectors6(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
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

// Outer -> inner ==> inner -> inner (nested)
extern "C" void addVectors7(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 3;
        int _occa_tiled_i = 0;
        outer[0] = ((entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int i = _occa_tiled_i;
        inner[2] = ((_occa_tiled_i + 4) - _occa_tiled_i + (1) - 1) / (1);
        int _occa_tiled_j = 0;
        inner[1] = ((entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int j = _occa_tiled_j;
        inner[0] = ((_occa_tiled_j + 4) - _occa_tiled_j + (1) - 1) / (1);
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> inner ==> inner -> inner (nested) + complex range + check true
extern "C" void addVectors8(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 3;
        int _occa_tiled_i = (entries - 12 + static_cast<int>(*a));
        outer[0] = (((entries + 16)) - ((entries - 12 + static_cast<int>(*a))) +
                    (4 * ((entries / 16 + 1))) - 1) /
                   (4 * ((entries / 16 + 1)));
        int i = _occa_tiled_i;
        inner[2] =
            ((_occa_tiled_i + 4) - _occa_tiled_i + ((entries / 16 + 1)) - 1) / ((entries / 16 + 1));
        unsigned long long _occa_tiled_j = (entries - 12 + static_cast<int>(*a));
        inner[1] = (((entries + 16)) - ((entries - 12 + static_cast<int>(*a))) +
                    (4 * ((entries / 16 + 1))) - 1) /
                   (4 * ((entries / 16 + 1)));
        unsigned long long j = _occa_tiled_j;
        inner[0] =
            ((_occa_tiled_j + 4) - _occa_tiled_j + ((entries / 16 + 1)) - 1) / ((entries / 16 + 1));
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}

// Outer -> inner, Outer -> inner
extern "C" void addVectors9(occa::modeKernel_t** deviceKernels,
                            const int& entries,
                            occa::modeMemory_t* a,
                            occa::modeMemory_t* b,
                            occa::modeMemory_t* ab) {
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[0] = (1 + (entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int i = _occa_tiled_i;
        inner[0] = ((_occa_tiled_i + 4) - _occa_tiled_i + (1) - 1) / (1);
        occa::kernel kernel(deviceKernels[0]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
    {
        occa::dim outer, inner;
        outer.dims = 1;
        inner.dims = 1;
        int _occa_tiled_i = 0;
        outer[0] = (1 + (entries) - (0) + (4 * (1)) - 1) / (4 * (1));
        int i = _occa_tiled_i;
        inner[0] = ((_occa_tiled_i + 4) - _occa_tiled_i + (1) - 1) / (1);
        occa::kernel kernel(deviceKernels[1]);
        kernel.setRunDims(outer, inner);
        kernel(entries, a, b, ab);
    };
}
