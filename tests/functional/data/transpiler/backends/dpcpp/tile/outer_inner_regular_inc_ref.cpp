#include <CL/sycl.hpp>
using namespace sycl;
const int offset = 1;

// template<typename T>
SYCL_EXTERNAL float add(float a, float b) { return a + b + offset; }

// Outer -> inner ==> regular -> regular
extern "C" [[sycl::reqd_work_group_size(1, 1, 4)]] void
_occa_addVectors0_0(sycl::queue *queue_, sycl::nd_range<3> *range_,
                    const int entries, const float *a, const float *b,
                    float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + (((4) * 1) * item_.get_group(2));
        {
          int i = _occa_tiled_i + ((1) * item.get_local_id(2));
          if (i < entries) {
            for (int _occa_tiled_j = 0; _occa_tiled_j < entries;
                 _occa_tiled_j += (4)) {
              for (int j = _occa_tiled_j; j < (_occa_tiled_j + (4)); ++j)
                if (j < entries) {
                  ab[i] = add(a[i], b[j]);
                }
            }
          }
        }
      }
    });
  });
}

// Outer -> inner ==> inner -> regular
extern "C" void _occa_addVectors2_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + (((4) * 1) * item_.get_group(2));
        {
          int i = _occa_tiled_i + ((1) * item.get_local_id(2));
          if (i < entries) {
            {
              int _occa_tiled_j = (0) + ((4) * item.get_local_id(1));
              for (int j = _occa_tiled_j; j < (_occa_tiled_j + (4)); ++j)
                if (j < entries) {
                  ab[i] = add(a[i], b[j]);
                }
            }
          }
        }
      }
    });
  });
}

// Outer -> inner ==> inner -> inner
extern "C" void _occa_addVectors3_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + (((4) * 1) * item_.get_group(2));
        {
          int i = _occa_tiled_i + ((1) * item.get_local_id(2));
          if (i < entries) {
            {
              int _occa_tiled_j = (0) + ((4) * item.get_local_id(1));
              {
                int j = _occa_tiled_j + item.get_local_id(1);
                if (j < entries) {
                  ab[i] = add(a[i], b[j]);
                }
              }
            }
          }
        }
      }
    });
  });
}

// Outer -> outer ==> inner -> regular
extern "C" void _occa_addVectors4_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + (((4) * 1) * item_.get_group(2));
        {
          int i = _occa_tiled_i + ((1) * item_.get_group(1));
          if (i < entries) {
            {
              int _occa_tiled_j = (0) + ((4) * item.get_local_id(1));
              for (int j = _occa_tiled_j; j < (_occa_tiled_j + (4)); ++j)
                if (j < entries) {
                  ab[i] = add(a[i], b[j]);
                }
            }
          }
        }
      }
    });
  });
}

// Outer -> outer ==> inner -> inner
extern "C" void _occa_addVectors5_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + (((4) * 1) * item_.get_group(2));
        {
          int i = _occa_tiled_i + ((1) * item_.get_group(1));
          if (i < entries) {
            {
              int _occa_tiled_j = (0) + ((4) * item.get_local_id(1));
              {
                int j = _occa_tiled_j + item.get_local_id(0);
                if (j < entries) {
                  ab[i] = add(a[i], b[j]);
                }
              }
            }
          }
        }
      }
    });
  });
}

// Outer -> outer ==> outer -> inner
extern "C" [[sycl::reqd_work_group_size(1, 1, 4)]] void
_occa_addVectors6_0(sycl::queue *queue_, sycl::nd_range<3> *range_,
                    const int entries, const float *a, const float *b,
                    float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + (((4) * 1) * item_.get_group(2));
        {
          int i = _occa_tiled_i + ((1) * item_.get_group(1));
          if (i < entries) {
            {
              int _occa_tiled_j = (0) + ((4) * item_.get_group(0));
              {
                int j = _occa_tiled_j + item.get_local_id(2);
                if (j < entries) {
                  ab[i] = add(a[i], b[j]);
                }
              }
            }
          }
        }
      }
    });
  });
}
