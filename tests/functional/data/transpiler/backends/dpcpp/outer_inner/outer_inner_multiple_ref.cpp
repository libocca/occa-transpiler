#include <CL/sycl.hpp>
using namespace sycl;
const int offset = 1;

// template<typename T>
SYCL_EXTERNAL float add(float a, float b) { return a + b + offset; }

// with shared memory usage (should be automatic sync)
extern "C" void _occa_addVectors_0(sycl::queue *queue_,
                                   sycl::nd_range<3> *range_, const int entries,
                                   float *a, float *b, float *ab, float *mat) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      int i = 0 + ((1) * item_.get_group(1));
      {
        int i2 = 0 + ((1) * item_.get_group(2));
        {
          auto &shm =
              *(sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
                  item_.get_group()));
          auto &shm2 =
              *(sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
                  item_.get_group()));
          {
            int j = 0 + ((1) * item.get_local_id(1));
            {
              shm[j] = 0; // shared memory usage -> should be barrier after
                          // @inner loop
              mat[0 + (10 * (0))] = 12;
              {
                int k = 0 + ((1) * item.get_local_id(1));
                {
                  {
                    int ii = 0 + ((1) * item.get_local_id(2));
                    { ab[i] = add(a[i], b[k]); }
                  }
                  ab[i] = add(a[i], b[k]);
                }
              }
              {
                int k = 0 + ((1) * item.get_local_id(1));
                {
                  {
                    int ii = 0 + ((1) * item.get_local_id(2));
                    { ab[i] = add(a[i], b[k]); }
                  }

                  ab[i] = add(a[i], b[k]);
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);
          }

          {
            int _occa_tiled_j = (0) + (((4) * 1) * item.get_local_id(0));
            {
              int j = _occa_tiled_j + ((1) * item.get_local_id(1));
              {
                {
                  int k = 0 + ((1) * item.get_local_id(2));
                  {
                    // shared memory usage -> should be barrier, since @tile is
                    // inner, inner
                    shm[j] = 0;
                  }
                }
              }
            }
            item_.barrier(sycl::access::fence_space::local_space);
          }

          {
            int j = 0 + ((1) * item.get_local_id(1));
            {
              shm[j] = 0;
              {
                int k = 0 + ((1) * item.get_local_id(1));
                {
                  {
                    int ii = 0 + ((1) * item.get_local_id(2));
                    { ab[i] = add(a[i], b[k]); }
                  }

                  ab[i] = add(a[i], b[k]);
                }
              }

              {
                int _occa_tiled_k = (0) + (((4) * 1) * item.get_local_id(1));
                {
                  int k = _occa_tiled_k + ((1) * item.get_local_id(2));
                  { ab[i] = add(a[i], b[k]); }
                }
              }
            }
          }
        }
      }
    });
  });
}

// without shared memory usage (should be no automatic sync)
extern "C" void _occa_addVectors1_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, float *a, float *b,
                                    float *ab, float *mat) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      int i = 0 + ((1) * item_.get_group(1));
      {
        int i2 = 0 + ((1) * item_.get_group(2));
        {
          auto &shm =
              *(sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
                  item_.get_group()));
          auto &shm2 =
              *(sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
                  item_.get_group()));
          {
            int j = 0 + ((1) * item.get_local_id(1));
            {
              // shm[j] = 0;  // shared memory usage -> should be barrier after
              // @inner loop
              mat[0 + (10 * (0))] = 12;
              {
                int k = 0 + ((1) * item.get_local_id(1));
                {
                  {
                    int ii = 0 + ((1) * item.get_local_id(2));
                    { ab[i] = add(a[i], b[k]); }
                  }
                  ab[i] = add(a[i], b[k]);
                }
              }
              {
                int k = 0 + ((1) * item.get_local_id(1));
                {
                  {
                    int ii = 0 + ((1) * item.get_local_id(2));
                    { ab[i] = add(a[i], b[k]); }
                  }

                  ab[i] = add(a[i], b[k]);
                }
              }
            }
          }

          {
            int _occa_tiled_j = (0) + (((4) * 1) * item.get_local_id(0));
            {
              int j = _occa_tiled_j + ((1) * item.get_local_id(1));
              {
                {
                  int k = 0 + ((1) * item.get_local_id(2));
                  {
                    // shared memory usage -> should be barrier, since @tile is
                    // inner, inner shm[j] = 0;
                  }
                }
              }
            }
          }

          {
            int j = 0 + ((1) * item.get_local_id(1));
            {
              shm[j] = 0;
              {
                int k = 0 + ((1) * item.get_local_id(1));
                {
                  {
                    int ii = 0 + ((1) * item.get_local_id(2));
                    { ab[i] = add(a[i], b[k]); }
                  }

                  ab[i] = add(a[i], b[k]);
                }
              }

              {
                int _occa_tiled_k = (0) + (((4) * 1) * item.get_local_id(1));
                {
                  int k = _occa_tiled_k + ((1) * item.get_local_id(2));
                  { ab[i] = add(a[i], b[k]); }
                }
              }
            }
          }
        }
      }
    });
  });
}