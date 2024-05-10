#include <CL/sycl.hpp>
using namespace sycl;

auto& sh_float32_t =
    *(sycl::ext::oneapi::group_local_memory_for_overwrite<typedef float>(item_.get_group()));

extern "C" [[sycl::reqd_work_group_size(1, 1, 32)]] void _occa_test_kernel_0(
    sycl::queue* queue_,
    sycl::nd_range<3>* range_) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            {
                int i = (0) + item_.get_group(2);
                auto& b = *(sycl::ext::oneapi::group_local_memory_for_overwrite<sh_float32_t[32]>(
                    item_.get_group()));
                {
                    int j = (0) + item.get_local_id(2);
                    b[j] = i + j;
                }
            }
        });
    });
}
