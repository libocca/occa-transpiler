#include <CL/sycl.hpp>
using namespace sycl;
// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void _occa_atomic_add_builtin_0(sycl::queue *queue_,
                                           sycl::nd_range<3> *range_,
                                           const int *iVec, int *iSum,
                                           const float *fVec, float *fSum) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(*iSum) +=
          iVec[0];
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(*fSum) +=
          fVec[0];
    });
  });
}

struct ComplexTypeF32 {
  float real;
  float imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void _occa_atomic_add_struct_0(sycl::queue *queue_,
                                          sycl::nd_range<3> *range_,
                                          const ComplexTypeF32 *vec,
                                          ComplexTypeF32 *sum) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(sum->real) +=
          vec[0].real;
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(sum->imag) +=
          vec[0].imag;
    });
  });
}

template <class T> struct ComplexType {
  T real;
  T imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" void _occa_atomic_add_template_0(sycl::queue *queue_,
                                            sycl::nd_range<3> *range_,
                                            const ComplexType<float> *vec,
                                            ComplexType<float> *sum) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(sum->real) +=
          vec[0].real;
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>(sum->imag) +=
          vec[0].imag;
    });
  });
}
