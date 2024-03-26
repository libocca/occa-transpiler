#include <CL/sycl.hpp>
using namespace sycl;

namespace A {
template <class T> struct Complex {
  T real;
  T imaginary;
};

namespace B {
struct Configs {
  unsigned int size1;
  unsigned long size2;
};

namespace C {
typedef int SIZE_TYPE;
typedef SIZE_TYPE SIZES;
} // namespace C
} // namespace B
} // namespace A


extern "C" [[sycl::reqd_work_group_size(1, 1, 1)]] void
_occa_function1_0(sycl::queue *queue_, sycl::nd_range<3> *range_,
                  const A::Complex<float> *__restrict__ vectorData,
                  unsigned int vectorSize,
                  const A::Complex<float> **__restrict__ matricesData,
                  const A::B::Configs *__restrict__ matricesSizes) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(2);
        { int j = (0) + item.get_local_id(2); }
      }
    });
  });
}


extern "C" [[sycl::reqd_work_group_size(1, 1, 1)]] void
_occa_function2_0(sycl::queue *queue_, sycl::nd_range<3> *range_,
                  const A::Complex<float> *__restrict__ vectorData,
                  const A::B::Configs *__restrict__ configs,
                  A::B::C::SIZES *__restrict__ vectorSize) {
  queue_->submit([&](sycl::handler &handler_) {
    handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
      {
        int i = (0) + item_.get_group(2);
        { int j = (0) + item.get_local_id(2); }
      }
    });
  });
}
