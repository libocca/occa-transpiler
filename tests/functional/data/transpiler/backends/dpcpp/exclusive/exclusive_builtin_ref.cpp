#include <CL/sycl.hpp>
using namespace sycl;

SYCL_EXTERNAL static float add(const float* a, int i, const float* b, int j) {
    return a[i] + b[j];
}

// TODO: fix preprocessor handling and try with define
// #define BLOCK_SIZE 4
const int BLOCK_SIZE = 4;

extern "C" [[sycl::reqd_work_group_size(1, 1, 4)]] void _occa_addVectors_0(
    sycl::queue* queue_,
    sycl::nd_range<3>* range_,
    const int N,
    const float* a,
    const float* b,
    float* ab) {
    queue_->submit([&](sycl::handler& handler_) {
        handler_.parallel_for(*range_, [=](sycl::nd_item<3> item_) {
            {
                int i = (0) + ((BLOCK_SIZE)*item_.get_group(2));
                auto& s_b = *(sycl::ext::oneapi::group_local_memory_for_overwrite<float[4]>(
                    item_.get_group()));
                const float* g_a = a;
                {
                    int j = (0) + item.get_local_id(2);
                    s_b[j] = b[i + j];
                    item_.barrier(sycl::access::fence_space::local_space) ab[i + j] =
                        add(g_a, i + j, s_b, j);
                }
            }
        });
    });
}
