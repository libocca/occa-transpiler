#include <CL/sycl.hpp>
using namespace sycl;

extern "C" [[sycl::reqd_work_group_size(1, 1, 10)]] void _occa_hello_kern_0(
    sycl::queue* queue_,
    sycl::nd_range<3>* range_) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            {
                int i = (0) + item_.get_group(2);
                auto& shm = *(sycl::ext::oneapi::group_local_memory_for_overwrite<int[10]>(
                    item_.get_group()));
                {
                    int j = (0) + item.get_local_id(2);
                    shm[j] = j;
                    item_.barrier(sycl::access::fence_space::local_space);
                }
                {
                    int j = (0) + item.get_local_id(2);
                    shm[j] = j;
                }
                {
                    int j = (0) + item.get_local_id(2);
                    shm[j] = j;
                    item_.barrier(sycl::access::fence_space::local_space);
                }
                {
                    int j = (0) + item.get_local_id(2);
                    shm[j] = j;
                }
            }
        });
    });
}
