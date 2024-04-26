#include <CL/sycl.hpp>
using namespace sycl;

extern "C" [[sycl::reqd_work_group_size(1, 1, 10)]] void _occa_test0_0(sycl::queue* queue_,
                                                                       sycl::nd_range<3>* range_) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_,
                              [=](sycl::nd_item<3> item_) [[intel::reqd_sub_group_size(16)]] {
                                  {
                                      int x = (0) + item_.get_group(2);
                                      {
                                          int y = (0) + item.get_local_id(2);
                                          int z = x + y;
                                      }
                                  }
                              });
    });
}

extern "C" [[sycl::reqd_work_group_size(1, 1, 10)]] void _occa_test1_0(sycl::queue* queue_,
                                                                       sycl::nd_range<3>* range_) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_,
                              [=](sycl::nd_item<3> item_) [[intel::reqd_sub_group_size(16)]] {
                                  {
                                      int _occa_tiled_x = (0) + ((5) * item_.get_group(2));
                                      for (int x = _occa_tiled_x; x < (_occa_tiled_x + (5)); ++x) {
                                          if (x < 10) {
                                              {
                                                  int y = (0) + item.get_local_id(2);
                                                  int z = x + y;
                                              }
                                          }
                                      }
                                  }
                              });
    });
}

extern "C" [[sycl::reqd_work_group_size(1, 1, 10)]] void _occa_test2_0(sycl::queue* queue_,
                                                                       sycl::nd_range<3>* range_) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_,
                              [=](sycl::nd_item<3> item_) [[intel::reqd_sub_group_size(16)]] {
                                  {
                                      int x = (0) + item_.get_group(2);
                                      {
                                          int y = (0) + item.get_local_id(2);
                                          int z = x + y;
                                      }
                                  }
                              });
    });
}

extern "C" [[sycl::reqd_work_group_size(1, 1, 10)]] void _occa_test2_1(sycl::queue* queue_,
                                                                       sycl::nd_range<3>* range_) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_,
                              [=](sycl::nd_item<3> item_) [[intel::reqd_sub_group_size(16)]] {
                                  {
                                      int x = (0) + item_.get_group(2);
                                      {
                                          int y = (0) + item.get_local_id(2);
                                          int z = x + y;
                                      }
                                  }
                              });
    });
}
