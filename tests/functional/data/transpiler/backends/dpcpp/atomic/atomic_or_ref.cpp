#include <CL/sycl.hpp>
using namespace sycl;
// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void _occa_atomic_and_builtin_0(sycl::queue *queue_,
                                           sycl::nd_range<3> *range_,
                                           const unsigned int *masks,
                                           unsigned int *mask) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(*mask) |=
          masks[0];
    });
  });
}

struct ComplexMaskType {
  unsigned int mask1;
  unsigned int mask2;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void _occa_atomic_and_struct_0(sycl::queue *queue_,
                                          sycl::nd_range<3> *range_,
                                          const ComplexMaskType *masks,
                                          ComplexMaskType *mask) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(
          mask->mask1) |= masks[0].mask1;
      sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(
          mask->mask2) |= masks[0].mask2;
    });
  });
}
