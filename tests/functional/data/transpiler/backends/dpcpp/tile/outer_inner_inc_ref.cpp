#include <CL/sycl.hpp>
using namespace sycl;

const int offset = 1;

// template<typename T>
SYCL_EXTERNAL float add(float a, float b) { return a + b + offset; }

// Outer -> inner
extern "C" void _occa_addVectors0_0(sycl::queue *queue_,
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
            ab[i] = add(a[i], b[i]);
          }
        }
      }
    });
  });
}

// Outer -> inner non 1 increment
extern "C" void _occa_addVectors1_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + (((4) * 2) * item_.get_group(2));
        {
          int i = _occa_tiled_i + ((2) * item.get_local_id(2));
          if (i < entries) {
            ab[i] = add(a[i], b[i]);
          }
        }
      }
    });
  });
}

// Outer -> inner unary post add
extern "C" void _occa_addVectors2_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + ((4) * item_.get_group(2));
        {
          int i = _occa_tiled_i + item.get_local_id(2);
          if (i < entries) {
            ab[i] = add(a[i], b[i]);
          }
        }
      }
    });
  });
}

// Outer -> inner unary pre add
extern "C" void _occa_addVectors3_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + ((4) * item_.get_group(2));
        {
          int i = _occa_tiled_i + item.get_local_id(2);
          if (i < entries) {
            ab[i] = add(a[i], b[i]);
          }
        }
      }
    });
  });
}

// Outer -> inner, check=True
extern "C" void _occa_addVectors4_0(sycl::queue *queue_,
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
            ab[i] = add(a[i], b[i]);
          }
        }
      }
    });
  });
}

// Outer -> inner, complex range
extern "C" void _occa_addVectors5_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = ((entries - 12 + 4)) +
                            (((4) * (entries / 16 + 1)) * item_.get_group(2));
        {
          int i = _occa_tiled_i + (((entries / 16 + 1)) * item.get_local_id(2));
          if (i < (entries + 16)) {
            ab[i] = add(a[i], b[i]);
          }
        }
      }
    });
  });
}

// Outer -> inner, set dimension
extern "C" void _occa_addVectors6_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + (((4) * 1) * item_.get_group(1));
        {
          int i = _occa_tiled_i + ((1) * item.get_local_id(0));
          if (i < entries) {
            ab[i] = add(a[i], b[i]);
          }
        }
      }
    });
  });
}

// Outer -> inner ==> inner -> inner (nested)
extern "C" void _occa_addVectors7_0(sycl::queue *queue_,
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
              int _occa_tiled_j = (0) + (((4) * 1) * item.get_local_id(1));
              {
                int j = _occa_tiled_j + ((1) * item.get_local_id(0));
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

// Outer -> inner ==> inner -> inner (nested) + complex range + check true
extern "C" void _occa_addVectors8_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = ((entries - 12 + static_cast<int>(*a))) +
                            (((4) * (entries / 16 + 1)) * item_.get_group(2));
        {
          int i = _occa_tiled_i + (((entries / 16 + 1)) * item.get_local_id(2));
          if (i < (entries + 16)) {
            {
              unsigned long long _occa_tiled_j =
                  ((entries - 12 + static_cast<int>(*a))) +
                  (((4) * (entries / 16 + 1)) * item.get_local_id(1));
              {
                unsigned long long j = _occa_tiled_j + (((entries / 16 + 1)) *
                                                        item.get_local_id(0));
                if (j < (entries + 16)) {
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

// Outer -> inner, <=
extern "C" void _occa_addVectors9_0(sycl::queue *queue_,
                                    sycl::nd_range<3> *range_,
                                    const int entries, const float *a,
                                    const float *b, float *ab) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int _occa_tiled_i = (0) + (((4) * 1) * item_.get_group(2));
        {
          int i = _occa_tiled_i + ((1) * item.get_local_id(2));
          if (i <= entries) {
            ab[i] = add(a[i], b[i]);
          }
        }
      }
    });
  });
}
