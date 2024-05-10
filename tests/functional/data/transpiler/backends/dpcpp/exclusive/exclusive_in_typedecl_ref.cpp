#include <CL/sycl.hpp>
using namespace sycl;

typedef float ex_float32_t;

extern "C" [[sycl::reqd_work_group_size(1, 1, 32)]] void _occa_test_kernel_0(
    sycl::queue* queue_,
    sycl::nd_range<3>* range_) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            {
                int i = (0) + item_.get_group(2);
                ex_float32_t d[32];
                {
                    int j = (0) + item.get_local_id(2);
                    d[j] = i - j;
                }
            }
        });
    });
}
