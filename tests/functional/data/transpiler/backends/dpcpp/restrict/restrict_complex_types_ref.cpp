#include <CL/sycl.hpp>
using namespace sycl;

template <class T>
struct Complex {
    T real;
    T imaginary;
};

struct Configs {
    unsigned int size1;
    unsigned long size2;
};

struct Data {
    float* __restrict__ x;
    float* __restrict__ y;
    unsigned long size;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void _occa_function1_0(sycl::queue* queue_,
                                  sycl::nd_range<3>* range_,
                                  const Complex<float>* __restrict__ vectorData,
                                  unsigned int vectorSize,
                                  const Complex<float>** __restrict__ matricesData,
                                  const Configs* __restrict__ matricesSizes) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {});
    });
}
