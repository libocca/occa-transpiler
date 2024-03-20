#include <CL/sycl.hpp>
using namespace sycl;

extern "C" [[sycl::reqd_work_group_size(1, 1, 10)]] void _occa_mykern_0(sycl::queue* queue_,
                                                                        sycl::nd_range<3>* range_,
                                                                        int aaa,
                                                                        int bbb) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            {
                int i = (0) + item_.get_group(2);
                {
                    int j = (0) + item.get_local_id(2);
                    // BODY
                }
            }
        });
    });
}
