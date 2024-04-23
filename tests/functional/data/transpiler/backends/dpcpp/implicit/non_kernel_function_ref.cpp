#include <CL/sycl.hpp>
using namespace sycl;

SYCL_EXTERNAL static float add1(const float *a, int i, const float *b, int j) {
  return a[i] + b[i];
}

SYCL_EXTERNAL float add2(const float *a, int i, const float *b, int j) {
  return a[i] + b[i];
}

// At least one @kern function is requried
extern "C" [[sycl::reqd_work_group_size(1, 1, 32)]] void
_occa_kern_0(sycl::queue *queue_, sycl::nd_range<3> *range_) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(2);
        { int j = (0) + item.get_local_id(2); }
      }
    });
  });
}
