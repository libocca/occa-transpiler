#include <cuda_runtime.h>
struct Coord {
    float x;
    float y;
};
typedef float* mat89_f;
typedef int* mat89_i;
typedef Coord* mat89_s;
typedef Coord* mat8_s;

// float dim
extern "C" __global__ void _occa_test_kernel_0_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat89_f mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            { ab[i] = a[i] + b[j] + mat[i + (8 * (j))]; }
        }
    }
}

// int dim
extern "C" __global__ void _occa_test_kernel_1_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat89_i mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            { ab[i] = a[i] + b[j] + static_cast<float>(mat[i + (8 * (j))]); }
        }
    }
}

// struct dim
extern "C" __global__ void _occa_test_kernel_2_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat89_s mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            { ab[i] = a[i] + b[j] + mat[i + (8 * (j))].x; }
        }
    }
}

// struct + single dim
extern "C" __global__ void _occa_test_kernel_3_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat8_s mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            { ab[i] = a[i] + b[j] + mat[i].x + mat[j].y; }
        }
    }
}

// inside attributed loop
// TODO: Update after rewriter conflict resolving is merged
// @kernel void test_kernel_4(const int entries, float* a, float* b, float* ab,
// mat89_s mat) {
//     for (int i = 0; i < mat(7, 7); i += mat(1, 1); @outer(0)) {
//         for (int j = mat(0, 0); j < entries; j += 1; @inner(0)) {
//             ab[i] = a[i] + b[j] + mat(i, j).x + mat(j, i).y;
//         }
//     }
// }

// assignment, comparison, etc.
extern "C" __global__ void _occa_test_kernel_5_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat89_s mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            {
                if (mat[i + (8 * (j))].x <= 0) {
                    mat[i + (8 * (j))].x =
                        a[i] + b[j] + mat[i + (8 * (j))].x + mat[j + (8 * (i))].y;
                }
            }
        }
    }
}

// nested mat
extern "C" __global__ void _occa_test_kernel_6_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat89_s mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            { mat[i + (8 * (j))] = a[i] + b[j] + mat[i + (8 * (mat[j + (8 * (0))]))]; }
        }
    }
}

__device__ float get1() {
    return 1;
}

// nested mat + complex expressions inside dim
extern "C" __global__ void _occa_test_kernel_7_0(const int entries,
                                                 float* a,
                                                 float* b,
                                                 float* ab,
                                                 mat89_s mat) {
    int i = (0) + ((1) * blockIdx.x);
    {
        {
            int j = (0) + ((1) * threadIdx.x);
            {
                mat[i + (8 * (j + get1() + (i * j / get1())))] =
                    a[i] + b[j] + mat[i + 12 + (8 * (mat[j + (8 * (get1()))]))];
            }
        }
    }
}
