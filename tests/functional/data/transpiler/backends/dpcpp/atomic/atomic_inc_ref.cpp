#include <CL/sycl.hpp>
using namespace sycl;


extern "C" [[sycl::reqd_work_group_size(1, 1, 1)]] void
_occa_atomic_inc_builtin_0(sycl::queue *queue_, sycl::nd_range<3> *range_,
                           unsigned int *value) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(2);
        {
          int j = (0) + item.get_local_id(2);
          sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(
              (*value))++;
        }
      }
    });
  });
}

struct ComplexMaskType {
  unsigned int val1;
  int val2;
};


extern "C" [[sycl::reqd_work_group_size(1, 1, 1)]] void
_occa_atomic_inc_struct_0(sycl::queue *queue_, sycl::nd_range<3> *range_,
                          ComplexMaskType *value) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(2);
        {
          int j = (0) + item.get_local_id(2);
          sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(
              value->val1)++;
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(
              value->val2)++;
        }
      }
    });
  });
}
