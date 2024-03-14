#include <hip/hip_runtime.h>
__constant__ int offset = 1;

// template<typename T>
__device__ float add(float a, float b) { return a + b + offset; }

// with shared memory usage (should be automatic sync)
extern "C" __global__ void _occa_addVectors_0(const int entries, float *a,
                                              float *b, float *ab, float *mat) {
  int i = (0) + ((1) * blockIdx.y);
  {
    int i2 = (0) + ((1) * blockIdx.x);
    {
      __shared__ int shm[32];
      __shared__ int shm2[32];
      {
        int j = (0) + ((1) * threadIdx.y);
        {
          shm[j] =
              0; // shared memory usage -> should be barrier after @inner loop
          mat[0 + (10 * (0))] = 12;
          {
            int k = (0) + ((1) * threadIdx.y);
            {
              {
                int ii = (0) + ((1) * threadIdx.x);
                { ab[i] = add(a[i], b[k]); }
              }
              ab[i] = add(a[i], b[k]);
            }
          }
          {
            int k = (0) + ((1) * threadIdx.y);
            {
              {
                int ii = (0) + ((1) * threadIdx.x);
                { ab[i] = add(a[i], b[k]); }
              }

              ab[i] = add(a[i], b[k]);
            }
          }
        }
        __syncthreads();
      }

      {
        int _occa_tiled_j = (0) + (((4) * 1) * threadIdx.z);
        {
          int j = _occa_tiled_j + ((1) * threadIdx.y);
          {
            {
              int k = (0) + ((1) * threadIdx.x);
              {
                // shared memory usage -> should be barrier, since @tile is
                // inner, inner
                shm[j] = 0;
              }
            }
          }
        }
        __syncthreads();
      }

      {
        int j = (0) + ((1) * threadIdx.y);
        {
          shm[j] = 0;
          {
            int k = (0) + ((1) * threadIdx.y);
            {
              {
                int ii = (0) + ((1) * threadIdx.x);
                { ab[i] = add(a[i], b[k]); }
              }

              ab[i] = add(a[i], b[k]);
            }
          }

          {
            int _occa_tiled_k = (0) + (((4) * 1) * threadIdx.y);
            {
              int k = _occa_tiled_k + ((1) * threadIdx.x);
              { ab[i] = add(a[i], b[k]); }
            }
          }
        }
      }
    }
  }
}

// without shared memory usage (should be no automatic sync)
extern "C" __global__ void _occa_addVectors1_0(const int entries, float *a,
                                               float *b, float *ab,
                                               float *mat) {
  int i = (0) + ((1) * blockIdx.y);
  {
    int i2 = (0) + ((1) * blockIdx.x);
    {
      __shared__ int shm[32];
      __shared__ int shm2[32];
      {
        int j = (0) + ((1) * threadIdx.y);
        {
          // shm[j] = 0;  // shared memory usage -> should be barrier after
          // @inner loop
          mat[0 + (10 * (0))] = 12;
          {
            int k = (0) + ((1) * threadIdx.y);
            {
              {
                int ii = (0) + ((1) * threadIdx.x);
                { ab[i] = add(a[i], b[k]); }
              }
              ab[i] = add(a[i], b[k]);
            }
          }
          {
            int k = (0) + ((1) * threadIdx.y);
            {
              {
                int ii = (0) + ((1) * threadIdx.x);
                { ab[i] = add(a[i], b[k]); }
              }

              ab[i] = add(a[i], b[k]);
            }
          }
        }
      }

      {
        int _occa_tiled_j = (0) + (((4) * 1) * threadIdx.z);
        {
          int j = _occa_tiled_j + ((1) * threadIdx.y);
          {
            {
              int k = (0) + ((1) * threadIdx.x);
              {
                // shared memory usage -> should be barrier, since @tile is
                // inner, inner shm[j] = 0;
              }
            }
          }
        }
      }

      {
        int j = (0) + ((1) * threadIdx.y);
        {
          shm[j] = 0;
          {
            int k = (0) + ((1) * threadIdx.y);
            {
              {
                int ii = (0) + ((1) * threadIdx.x);
                { ab[i] = add(a[i], b[k]); }
              }

              ab[i] = add(a[i], b[k]);
            }
          }

          {
            int _occa_tiled_k = (0) + (((4) * 1) * threadIdx.y);
            {
              int k = _occa_tiled_k + ((1) * threadIdx.x);
              { ab[i] = add(a[i], b[k]); }
            }
          }
        }
      }
    }
  }
}