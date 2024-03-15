#include <occa/core/kernel.hpp>

const int offset = 1;

// template<typename T>
float add(float a, float b) { return a + b + offset; }

// Outer -> inner
extern "C" void addVectors0(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
  {
    occa::dim outer, inner;
    outer.dims = 1;
    inner.dims = 1;
    int j = 0;
    outer[0] = ((entries) - (0) + (1) - 1) / (1);
    int i = 0;
    inner[0] = ((entries) - (0) + (1) - 1) / (1);
    occa::kernel kernel(deviceKernels[0]);
    kernel.setRunDims(outer, inner);
    kernel(entries, a, b, ab);
  };
}

// Outer -> inner non 1 increment
extern "C" void addVectors1(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
  {
    occa::dim outer, inner;
    outer.dims = 1;
    inner.dims = 1;
    int j = 0;
    outer[0] = ((entries) - (0) + (2) - 1) / (2);
    int i = 0;
    inner[0] = ((entries) - (0) + (2) - 1) / (2);
    occa::kernel kernel(deviceKernels[0]);
    kernel.setRunDims(outer, inner);
    kernel(entries, a, b, ab);
  };
}

// Outer -> inner unary post add
extern "C" void addVectors2(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
  {
    occa::dim outer, inner;
    outer.dims = 1;
    inner.dims = 1;
    int j = 0;
    outer[0] = (entries) - (0);
    int i = 0;
    inner[0] = 1 + (entries - 1) - (0);
    occa::kernel kernel(deviceKernels[0]);
    kernel.setRunDims(outer, inner);
    kernel(entries, a, b, ab);
  };
}

// Outer -> inner unary pre add
extern "C" void addVectors3(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
  {
    occa::dim outer, inner;
    outer.dims = 1;
    inner.dims = 1;
    int j = 0;
    outer[0] = (entries) - (0);
    int i = 0;
    inner[0] = (entries) - (0);
    occa::kernel kernel(deviceKernels[0]);
    kernel.setRunDims(outer, inner);
    kernel(entries, a, b, ab);
  };
}

// Outer -> outer -> inner -> inner
// TODO: change after sema calculates dimensions
extern "C" void addVectors4(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
  {
    occa::dim outer, inner;
    outer.dims = 2;
    inner.dims = 0;
    int i = 0;
    outer[1] = (entries) - (0);
    int j = 0;
    outer[0] = (entries) - (0);
    occa::kernel kernel(deviceKernels[0]);
    kernel.setRunDims(outer, inner);
    kernel(entries, a, b, ab);
  };
}

// Outer -> outer -> inner -> inner + manual dimensions specification
extern "C" void addVectors5(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
  {
    occa::dim outer, inner;
    outer.dims = 2;
    inner.dims = 0;
    int i = 0;
    outer[1] = (entries) - (0);
    int j = 0;
    outer[0] = (entries) - (0);
    occa::kernel kernel(deviceKernels[0]);
    kernel.setRunDims(outer, inner);
    kernel(entries, a, b, ab);
  };
}

// Outer -> outer -> inner -> inner + partially manual dimensions specification
// TODO: change after sema calculates dimensions
extern "C" void addVectors6(occa::modeKernel_t **deviceKernels,
                            const int &entries, occa::modeMemory_t *a,
                            occa::modeMemory_t *b, occa::modeMemory_t *ab) {
  {
    occa::dim outer, inner;
    outer.dims = 2;
    inner.dims = 0;
    int i = 0;
    outer[1] = (entries) - (0);
    int j = 0;
    outer[0] = (entries) - (0);
    occa::kernel kernel(deviceKernels[0]);
    kernel.setRunDims(outer, inner);
    kernel(entries, a, b, ab);
  };
}
