#include <CL/sycl.hpp>
using namespace sycl;
const int offset = 1;

// template<typename T>
SYCL_EXTERNAL float add(float a, float b) {
    return a + b + offset;
}

// Outer -> inner
extern "C" void _occa_addVectors0_0(sycl::queue* queue_,
                                    sycl::nd_range<3>* range_,
                                    const int entries,
                                    const float* a,
                                    const float* b,
                                    float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            int j = 0 + ((1) * item_.get_group(2));
            {
                int i = 0 + ((1) * item.get_local_id(2));
                { ab[i] = add(a[i], b[i]); }
            }
        });
    });
}

// Outer -> inner non 1 increment
extern "C" void _occa_addVectors1_0(sycl::queue* queue_,
                                    sycl::nd_range<3>* range_,
                                    const int entries,
                                    const float* a,
                                    const float* b,
                                    float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            int j = 0 + ((2) * item_.get_group(2));
            {
                int i = 0 + ((2) * item.get_local_id(2));
                { ab[i] = add(a[i], b[i]); }
            }
        });
    });
}

// Outer -> inner unary post add
extern "C" void _occa_addVectors2_0(sycl::queue* queue_,
                                    sycl::nd_range<3>* range_,
                                    const int entries,
                                    const float* a,
                                    const float* b,
                                    float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            int j = 0 + item_.get_group(2);
            {
                int i = 0 + item.get_local_id(2);
                { ab[i] = add(a[i], b[i]); }
            }
        });
    });
}

// Outer -> inner unary pre add
extern "C" void _occa_addVectors3_0(sycl::queue* queue_,
                                    sycl::nd_range<3>* range_,
                                    const int entries,
                                    const float* a,
                                    const float* b,
                                    float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            int j = 0 + item_.get_group(2);
            {
                int i = 0 + item.get_local_id(2);
                { ab[i] = add(a[i], b[i]); }
            }
        });
    });
}

// Outer -> outer -> inner -> inner
// TODO: change after sema calculates dimensions
extern "C" void _occa_addVectors4_0(sycl::queue* queue_,
                                    sycl::nd_range<3>* range_,
                                    const int entries,
                                    const float* a,
                                    const float* b,
                                    float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            int i = 0 + item_.get_group(2);
            {
                int j = 0 + item_.get_group(2);
                {
                    int k = 0 + item.get_local_id(2);
                    {
                        int ii = 0 + item.get_local_id(2);
                        { ab[ii + k] = add(a[i], b[j]); }
                    }
                }
            }
        });
    });
}

// Outer -> outer -> inner -> inner + manual dimensions specification
extern "C" void _occa_addVectors5_0(sycl::queue* queue_,
                                    sycl::nd_range<3>* range_,
                                    const int entries,
                                    const float* a,
                                    const float* b,
                                    float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            int i = 0 + item_.get_group(1);
            {
                int j = 0 + item_.get_group(2);
                {
                    int k = 0 + item.get_local_id(1);
                    {
                        int ii = 0 + item.get_local_id(2);
                        { ab[ii + k] = add(a[i], b[j]); }
                    }
                }
            }
        });
    });
}

// Outer -> outer -> inner -> inner + partially manual dimensions specification
// TODO: change after sema calculates dimensions
extern "C" void _occa_addVectors6_0(sycl::queue* queue_,
                                    sycl::nd_range<3>* range_,
                                    const int entries,
                                    const float* a,
                                    const float* b,
                                    float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            int i = 0 + item_.get_group(2);
            {
                int j = 0 + item_.get_group(2);
                {
                    int k = 0 + item.get_local_id(2);
                    {
                        int ii = 0 + item.get_local_id(2);
                        { ab[ii + k] = add(a[i], b[j]); }
                    }
                }
            }
        });
    });
}
