#include <CL/sycl.hpp>
using namespace sycl;

extern "C" [[sycl::reqd_work_group_size(1, 1, 10)]] void
_occa_hello_kern_0(sycl::queue *queue_, sycl::nd_range<3> *range_) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(2);
        auto &shm =
            *(sycl::ext::oneapi::group_local_memory_for_overwrite<int[10]>(
                item_.get_group()));
        {
          int j = (0) + item.get_local_id(2);
          shm[j] = j;
        }
        item_.barrier(sycl::access::fence_space::local_space);
        {
          int j = (0) + item.get_local_id(2);
          shm[j] = j;
        }
        {
          int j = (0) + item.get_local_id(2);
          shm[j] = j;
        }
        item_.barrier(sycl::access::fence_space::local_space);
        {
          int j = (0) + item.get_local_id(2);
          shm[j] = j;
        }
      }
    });
  });
}

extern "C" [[sycl::reqd_work_group_size(1, 1, 32)]] void
_occa_priority_issue_0(sycl::queue *queue_, sycl::nd_range<3> *range_) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(2);
        auto &shm =
            *(sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
                item_.get_group()));
        {
          int j = (0) + item.get_local_id(2);
          shm[i] = i;
        }
        {
          int j = (0) + item.get_local_id(2);
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(
              shm[i * j]) += 32;
        }
      }
    });
  });
}
