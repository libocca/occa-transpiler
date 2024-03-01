#include <CL/sycl.hpp>
using namespace sycl;

template <class T> struct ComplexType {
  T real;
  T imaginary;
};

// TODO: fix me when @kernel/@outer/@inner will be implementeds
extern "C" void _occa_function1_0(sycl::queue *queue_,
                                  sycl::nd_range<3> *range_, const int *data) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      auto &arr1 = *(sycl::ext::oneapi::group_local_memory_for_overwrite<
                     ComplexType<int>[32]>(item_.get_group()));
      auto &arr2 = *(sycl::ext::oneapi::group_local_memory_for_overwrite<
                     ComplexType<float>[8][32]>(item_.get_group()));
    });
  });
}
