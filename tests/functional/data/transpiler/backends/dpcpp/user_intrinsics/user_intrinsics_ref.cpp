#include <CL/sycl.hpp>

// INFO: from documentation
//  isNaN
//  Description:The function returns 1, if and only if its argument is a NaN.
//  Calling interface:
//  int __binary32_isNaN(float x);
SYCL_EXTERNAL bool okl_is_nan(float value) {
  return __binary32_isNaN(value) == 1;
}

using namespace sycl;

extern "C" [[sycl::reqd_work_group_size(1, 1, 32)]] void
_occa_zero_nans_0(sycl::queue *queue_, sycl::nd_range<3> *range_, float *vec) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(2);
        {
          int j = (0) + item.get_local_id(2);
          int idx = i * 32 + j;
          float value = vec[idx];
          if (okl_is_nan(value)) {
            vec[idx] = 0.0f;
          }
        }
      }
    });
  });
}

