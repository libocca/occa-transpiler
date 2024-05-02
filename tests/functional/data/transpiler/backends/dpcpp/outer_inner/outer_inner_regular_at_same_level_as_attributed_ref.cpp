#include <CL/sycl.hpp>
using namespace sycl;

extern "C" [[sycl::reqd_work_group_size(1, 1, 10)]] void
_occa_test_kernel_0(sycl::queue *queue_, sycl::nd_range<3> *range_) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(1);
        {
          int i2 = (0) + item_.get_group(2);
          { int j = (0) + item.get_local_id(2); }
          for (int ii = 0; ii < 10; ++ii) {
            {
              int j = (0) + item.get_local_id(2);
            }
            for (int j = 0; j < 10; ++j) {
            }
          }
        }
        for (int ii = 0; ii < 10; ++ii) {
          {
            int i = (0) + item_.get_group(2);
            { int j = (0) + item.get_local_id(2); }
          }
        }
      }
    });
  });
}

extern "C" [[sycl::reqd_work_group_size(1, 1, 10)]] void
_occa_test_kernel_1(sycl::queue *queue_, sycl::nd_range<3> *range_) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(1);
        for (int i2 = 0; i2 < 10; ++i2) {
          {
            int i2 = (0) + item_.get_group(2);
            { int j = (0) + item.get_local_id(2); }
          }
        }
      }
    });
  });
}
