#include <CL/sycl.hpp>
using namespace sycl;

struct ComplexTypeF32 {
  ComplexTypeF32 &operator=(const ComplexTypeF32 &) = default;
  float real;
  float imag;
};

// TODO: fix me when @kernel/@outer/@inner are implemented
extern "C" [[sycl::reqd_work_group_size(1, 1, 1)]] void
_occa_atomic_exch_struct_0(sycl::queue *queue_, sycl::nd_range<3> *range_,
                           const ComplexTypeF32 *vec, ComplexTypeF32 *result) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(2);
        {
          int j = (0) + item.get_local_id(2);
          sycl::atomic_ref<ComplexTypeF32, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(*result) =
              vec[0];
        }
      }
    });
  });
}
