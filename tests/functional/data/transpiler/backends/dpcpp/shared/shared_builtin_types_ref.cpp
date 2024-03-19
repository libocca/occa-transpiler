#include <CL/sycl.hpp>
using namespace sycl;

extern "C" [[sycl::reqd_work_group_size(1, 1, 64)]] void
_occa_function1_0(sycl::queue *queue_, sycl::nd_range<3> *range_,
                  const int *data) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      int i = (0) + item_.get_group(2);
      {
        auto &arr1 =
            *(sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
                item_.get_group()));
        auto &arr2 =
            *(sycl::ext::oneapi::group_local_memory_for_overwrite<float[8][32]>(
                item_.get_group()));
        auto &arr3 =
            *(sycl::ext::oneapi::group_local_memory_for_overwrite<double[8]>(
                item_.get_group()));
        {
          int j = (0) + item.get_local_id(2);
          {}
        }
      }
    });
  });
}
