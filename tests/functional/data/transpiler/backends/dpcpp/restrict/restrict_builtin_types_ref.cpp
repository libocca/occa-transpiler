#include <CL/sycl.hpp>
using namespace sycl;

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void _occa_function1_0(sycl::queue *queue_,
                                  sycl::nd_range<3> *range_,
                                  const int *__restrict__ i32Data,
                                  float *__restrict__ fp32Data,
                                  const double *__restrict__ fp64Data) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {});
  });
}
