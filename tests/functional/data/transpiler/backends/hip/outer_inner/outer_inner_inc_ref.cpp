#include <hip/hip_runtime.h>
__constant__ int offset = 1;

// template<typename T>
__device__ float add(float a, float b) { return a + b + offset; }

// Outer -> inner
extern "C" __global__ void _occa_addVectors0_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
    {
        int j = (0) + ((1) * blockIdx.x);
        {
      int i = (0) + ((1) * threadIdx.x);
            ab[i] = add(a[i], b[i]);
        }
  }
}

// Outer -> inner non 1 increment
extern "C" __global__ void _occa_addVectors1_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
    {
        int j = (0) + ((2) * blockIdx.x);
        {
      int i = (0) + ((2) * threadIdx.x);
            ab[i] = add(a[i], b[i]);
        }
  }
}

// Outer -> inner unary post add
extern "C" __global__ void _occa_addVectors2_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
    {
        int j = (0) + blockIdx.x;
        {
      int i = (0) + threadIdx.x;
            ab[i] = add(a[i], b[i]);
        }
  }
}

// Outer -> inner unary pre add
extern "C" __global__ void _occa_addVectors3_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
    {
        int j = (0) + blockIdx.x;
        {
      int i = (0) + threadIdx.x;
            ab[i] = add(a[i], b[i]);
        }
  }
}

// Outer -> outer -> inner -> inner
// TODO: change after sema calculates dimensions
extern "C" __global__ void _occa_addVectors4_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
    {
        int i = (0) + blockIdx.y;
        {
            int j = (0) + blockIdx.x;
            {
        int k = (0) + threadIdx.y;
        {
            int ii = (0) + threadIdx.x;
            ab[ii + k] = add(a[i], b[j]);
        }
      }
    }
  }
}

// Outer -> outer -> inner -> inner + manual dimensions specification
extern "C" __global__ void _occa_addVectors5_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
    {
        int i = (0) + blockIdx.y;
        {
            int j = (0) + blockIdx.x;
            {
        int k = (0) + threadIdx.y;
        {
            int ii = (0) + threadIdx.x;
            ab[ii + k] = add(a[i], b[j]);
        }
      }
    }
  }
}

// Outer -> outer -> inner -> inner + partially manual dimensions specification
// TODO: change after sema calculates dimensions
extern "C" __global__ void _occa_addVectors6_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
    {
        int i = (0) + blockIdx.y;
        {
            int j = (0) + blockIdx.x;
            {
        int k = (0) + threadIdx.y;
        {
            int ii = (0) + threadIdx.x;
            ab[ii + k] = add(a[i], b[j]);
        }
      }
    }
  }
}
