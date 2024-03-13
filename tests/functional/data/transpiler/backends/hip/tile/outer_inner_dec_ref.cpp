#include <hip/hip_runtime.h>
__constant__ int offset = 1;

// template<typename T>
__device__ float add(float a, float b) { return a + b + offset; }

// Outer -> inner
extern "C" __global__ void _occa_addVectors0_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i = ((entries - 1)) - (((4) * 1) * blockIdx.x);
    {
      int i = _occa_tiled_i - ((1) * threadIdx.x);
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner non 1 increment
extern "C" __global__ void _occa_addVectors1_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i = ((entries - 1)) - (((4) * 2) * blockIdx.x);
    {
      int i = _occa_tiled_i - ((2) * threadIdx.x);
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner unary post add
extern "C" __global__ void _occa_addVectors2_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i = ((entries - 1)) - ((4) * blockIdx.x);
    {
      int i = _occa_tiled_i - threadIdx.x;
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner unary pre add
extern "C" __global__ void _occa_addVectors3_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i = ((entries - 1)) - ((4) * blockIdx.x);
    {
      int i = _occa_tiled_i - threadIdx.x;
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner, check=True
extern "C" __global__ void _occa_addVectors4_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i = ((entries - 1)) - (((4) * 1) * blockIdx.x);
    {
      int i = _occa_tiled_i - ((1) * threadIdx.x);
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner, complex range
extern "C" __global__ void _occa_addVectors5_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i =
        ((entries + 16)) - (((4) * (entries / 16 + 1)) * blockIdx.x);
    {
      int i = _occa_tiled_i - (((entries / 16 + 1)) * threadIdx.x);
      if (i >= (entries - 12 + 4)) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner, set dimension
extern "C" __global__ void _occa_addVectors6_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i = ((entries - 1)) - (((4) * 1) * blockIdx.y);
    {
      int i = _occa_tiled_i - ((1) * threadIdx.z);
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner ==> inner -> inner (nested)
extern "C" __global__ void _occa_addVectors7_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i = ((entries - 1)) - (((4) * 1) * blockIdx.x);
    {
      int i = _occa_tiled_i - ((1) * threadIdx.x);
      if (i >= 0) {
        {
          int _occa_tiled_j = ((entries - 1)) - (((4) * 1) * threadIdx.y);
          {
            int j = _occa_tiled_j - ((1) * threadIdx.z);
            if (j >= 0) {
              ab[i] = add(a[i], b[j]);
            }
          }
        }
      }
    }
  }
}

// Outer -> inner ==> inner -> inner (nested) + complex range + check true
extern "C" __global__ void _occa_addVectors8_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i =
        ((entries + 16)) - (((4) * (entries / 16 + 1)) * blockIdx.x);
    {
      int i = _occa_tiled_i - (((entries / 16 + 1)) * threadIdx.x);
      if (i >= (entries - 12 + static_cast<int>(*a))) {
        {
          unsigned long long _occa_tiled_j =
              ((entries + 16)) - (((4) * (entries / 16 + 1)) * threadIdx.y);
          {
            unsigned long long j =
                _occa_tiled_j - (((entries / 16 + 1)) * threadIdx.z);
            if (j >= (entries - 12 + static_cast<int>(*a))) {
              ab[i] = add(a[i], b[j]);
            }
          }
        }
      }
    }
  }
}

// Outer -> inner ==> inner -> inner (nested) + complex range + check true +
// automatic dim claculation
extern "C" __global__ void _occa_addVectors9_0(const int entries,
                                               const float *a, const float *b,
                                               float *ab) {
  {
    int _occa_tiled_i =
        ((entries + 16)) - (((4) * (entries / 16 + 1)) * blockIdx.x);
    {
      int i = _occa_tiled_i - (((entries / 16 + 1)) * threadIdx.z);
      if (i >= (entries - 12 + static_cast<int>(*a))) {
        {
          unsigned long long _occa_tiled_j =
              ((entries + 16)) - (((4) * (entries / 16 + 1)) * threadIdx.y);
          {
            unsigned long long j =
                _occa_tiled_j - (((entries / 16 + 1)) * threadIdx.x);
            if (j >= (entries - 12 + static_cast<int>(*a))) {
              ab[i] = add(a[i], b[j]);
            }
          }
        }
      }
    }
  }
}
