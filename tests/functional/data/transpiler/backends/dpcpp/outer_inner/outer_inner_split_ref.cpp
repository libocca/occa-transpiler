#include <CL/sycl.hpp>
using namespace sycl;

extern "C" [[sycl::reqd_work_group_size(2, 5, 7)]] void _occa_test0_0(sycl::queue* queue_,
                                                                      sycl::nd_range<3>* range_,
                                                                      const int entries,
                                                                      const float* a,
                                                                      const float* b,
                                                                      float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            {
                int x = (0) + item_.get_group(0);
                // int before1 = 1 + before0;
                int before1 = 1;
                {
                    int y = (0) + item_.get_group(1);
                    int before2 = 1 + before1;
                    {
                        int z = (0) + item_.get_group(2);
                        int before3 = 1 + before2;
                        {
                            int n = (0) + item.get_local_id(0);
                            int after0 = 1 + before3;
                            {
                                int m = (0) + item.get_local_id(1);
                                int after1 = 1 + after0;
                                {
                                    int k = (0) + item.get_local_id(2);
                                    int after2 = 1 + after1;
                                    ab[x] = a[x] + b[x] +
                                            static_cast<float>(k + m + n + z + y + x + after2);
                                }
                                {
                                    int k = (0) + item.get_local_id(2);
                                    int after2 = 1 + after1;
                                    ab[x] = a[x] + b[x] +
                                            static_cast<float>(k + m + n + z + y + x + after2);
                                }
                            }
                            {
                                int m = (0) + item.get_local_id(1);
                                int after1 = 1 + after0;
                                {
                                    int k = (0) + item.get_local_id(2);
                                    int after2 = 1 + after1;
                                    ab[x] = a[x] + b[x] +
                                            static_cast<float>(k + m + n + z + y + x + after2);
                                }
                                {
                                    int k = (0) + item.get_local_id(2);
                                    int after2 = 1 + after1;
                                    ab[x] = a[x] + b[x] +
                                            static_cast<float>(k + m + n + z + y + x + after2);
                                }
                            }
                        }
                    }
                }
            }
        });
    });
}

extern "C" [[sycl::reqd_work_group_size(2, 5, 7)]] void _occa_test0_1(sycl::queue* queue_,
                                                                      sycl::nd_range<3>* range_,
                                                                      const int entries,
                                                                      const float* a,
                                                                      const float* b,
                                                                      float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            {
                int x = (0) + item_.get_group(0);
                // int before1 = 1 + before00;
                int before1 = 1;
                {
                    int y = (0) + item_.get_group(1);
                    int before2 = 1 + before1;
                    {
                        int z = (0) + item_.get_group(2);
                        int before3 = 1 + before2;
                        {
                            int n = (0) + item.get_local_id(0);
                            int after0 = 1 + before3;
                            {
                                int m = (0) + item.get_local_id(1);
                                int after1 = 1 + after0;
                                {
                                    int k = (0) + item.get_local_id(2);
                                    int after2 = 1 + after1;
                                    ab[x] = a[x] + b[x] +
                                            static_cast<float>(k + m + n + z + y + x + after2);
                                }
                                {
                                    int k = (0) + item.get_local_id(2);
                                    int after2 = 1 + after1;
                                    ab[x] = a[x] + b[x] +
                                            static_cast<float>(k + m + n + z + y + x + after2);
                                }
                            }
                            {
                                int m = (0) + item.get_local_id(1);
                                int after1 = 1 + after0;
                                {
                                    int k = (0) + item.get_local_id(2);
                                    int after2 = 1 + after1;
                                    ab[x] = a[x] + b[x] +
                                            static_cast<float>(k + m + n + z + y + x + after2);
                                }
                                {
                                    int k = (0) + item.get_local_id(2);
                                    int after2 = 1 + after1;
                                    ab[x] = a[x] + b[x] +
                                            static_cast<float>(k + m + n + z + y + x + after2);
                                }
                            }
                        }
                    }
                }
            }
        });
    });
}
