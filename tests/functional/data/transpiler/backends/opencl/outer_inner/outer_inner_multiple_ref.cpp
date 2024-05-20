#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__constant int offset = 1;
// template<typename T>
float add(float a, float b);

float add(float a, float b) { return a + b + offset; }

// with shared memory usage (should be automatic sync)
__kernel void _occa_addVectors_0(const int entries, __global float *a,
                                 __global float *b, __global float *ab,
                                 __global float *mat);

__kernel void _occa_addVectors_0(const int entries, __global float *a,
                                 __global float *b, __global float *ab,
                                 __global float *mat) {
  {
    int i = (0) + ((1) * get_group_id(1));
    {
      int i2 = (0) + ((1) * get_group_id(0));
      __local int shm[32];
      __local int shm2[32];
      {
        int j = (0) + ((1) * get_local_id(2));
        shm[j] =
            0; // shared memory usage -> should be barrier after @inner loop
        mat[0 + (10 * (0))] = 12;
        {
          int k = (0) + ((1) * get_local_id(1));
          {
            int ii = (0) + ((1) * get_local_id(0));
            ab[i] = add(a[i], b[k]);
          }
          ab[i] = add(a[i], b[k]);
        }
        {
          int k = (0) + ((1) * get_local_id(1));
          {
            int ii = (0) + ((1) * get_local_id(0));
            ab[i] = add(a[i], b[k]);
          }
          ab[i] = add(a[i], b[k]);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      {
        int _occa_tiled_j = (0) + (((4) * 1) * get_local_id(2));
        {
          int j = _occa_tiled_j + ((1) * get_local_id(1));
          {
            {
              int k = (0) + ((1) * get_local_id(0));
              // shared memory usage -> should be barrier, since @tile is inner,
              // inner
              shm[j] = 0;
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      {
        int j = (0) + ((1) * get_local_id(2));
        shm[j] = 0;
        {
          int k = (0) + ((1) * get_local_id(1));
          {
            int ii = (0) + ((1) * get_local_id(0));
            ab[i] = add(a[i], b[k]);
          }
          ab[i] = add(a[i], b[k]);
        }
        {
          int _occa_tiled_k = (0) + (((4) * 1) * get_local_id(1));
          {
            int k = _occa_tiled_k + ((1) * get_local_id(0));
            { ab[i] = add(a[i], b[k]); }
          }
        }
      }
    }
  }
}

// without shared memory usage (should be no automatic sync)
__kernel void _occa_addVectors1_0(const int entries, __global float *a,
                                  __global float *b, __global float *ab,
                                  __global float *mat);

__kernel void _occa_addVectors1_0(const int entries, __global float *a,
                                  __global float *b, __global float *ab,
                                  __global float *mat) {
  {
    int i = (0) + ((1) * get_group_id(1));
    {
      int i2 = (0) + ((1) * get_group_id(0));
      __local int shm[32];
      __local int shm2[32];
      {
        int j = (0) + ((1) * get_local_id(2));
        // shm[j] = 0; // shared memory usage -> should be barrier after @inner
        // loop
        mat[0 + (10 * (0))] = 12;
        {
          int k = (0) + ((1) * get_local_id(1));
          {
            int ii = (0) + ((1) * get_local_id(0));
            ab[i] = add(a[i], b[k]);
          }
          ab[i] = add(a[i], b[k]);
        }
        {
          int k = (0) + ((1) * get_local_id(1));
          {
            int ii = (0) + ((1) * get_local_id(0));
            ab[i] = add(a[i], b[k]);
          }
          ab[i] = add(a[i], b[k]);
        }
      }
      {
        int _occa_tiled_j = (0) + (((4) * 1) * get_local_id(2));
        {
          int j = _occa_tiled_j + ((1) * get_local_id(1));
          {
            {
              int k = (0) + ((1) * get_local_id(0));
              // shared memory usage -> should be barrier, since @tile is inner,
              // inner shm[j] = 0;
            }
          }
        }
      }
      {
        int j = (0) + ((1) * get_local_id(2));
        shm[j] = 0;
        {
          int k = (0) + ((1) * get_local_id(1));
          {
            int ii = (0) + ((1) * get_local_id(0));
            ab[i] = add(a[i], b[k]);
          }
          ab[i] = add(a[i], b[k]);
        }
        {
          int _occa_tiled_k = (0) + (((4) * 1) * get_local_id(1));
          {
            int k = _occa_tiled_k + ((1) * get_local_id(0));
            { ab[i] = add(a[i], b[k]); }
          }
        }
      }
    }
  }
}
