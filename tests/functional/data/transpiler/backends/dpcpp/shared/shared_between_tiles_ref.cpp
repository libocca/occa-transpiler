#include <CL/sycl.hpp>
using namespace sycl;

extern "C" void _occa_test_kern_0(sycl::queue *queue_,
                                  sycl::nd_range<3> *range_) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + ((4) * item_.get_group(2));
        for (int i = _occa_tiled_i; i < (_occa_tiled_i + (4)); ++i) {
          if (i < 10) {
            auto &shm =
                *(sycl::ext::oneapi::group_local_memory_for_overwrite<int[10]>(
                    item_.get_group()));
            {
              int _occa_tiled_j = (0) + ((4) * item.get_local_id(1));
              {
                int j = _occa_tiled_j + item.get_local_id(2);
                if (j < 10) {
                  shm[j] = j;
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);
          }
        }
      }
    });
  });
}
