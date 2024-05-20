#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__constant int offset = 1;
float add(float a, float b);

float add(float a, float b) { return a + b + offset; }

// Outer -> inner ==> regular -> regular
__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors0_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors0_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 1) * get_group_id(0));
    {
      int i = _occa_tiled_i - ((1) * get_local_id(0));
      if (i >= 0) {
        for (int _occa_tiled_j = entries; _occa_tiled_j > 0;
             _occa_tiled_j -= (4)) {
          for (int j = _occa_tiled_j; j > (_occa_tiled_j - (4)); --j) {
            if (j > 0) {
              ab[i] = add(a[i], b[j - 1]);
            }
          }
        }
      }
    }
  }
}

// Outer -> inner ==> inner -> regular
__kernel void _occa_addVectors2_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors2_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 1) * get_group_id(0));
    {
      int i = _occa_tiled_i - ((1) * get_local_id(0));
      if (i >= 0) {
        {
          int _occa_tiled_j = (entries) - ((4) * get_local_id(1));
          for (int j = _occa_tiled_j; j > (_occa_tiled_j - (4)); --j) {
            if (j > 0) {
              ab[i] = add(a[i], b[j - 1]);
            }
          }
        }
      }
    }
  }
}

// Outer -> inner ==> inner -> inner
__kernel void _occa_addVectors3_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors3_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 1) * get_group_id(0));
    {
      int i = _occa_tiled_i - ((1) * get_local_id(0));
      if (i >= 0) {
        {
          int _occa_tiled_j = (entries) - ((4) * get_local_id(1));
          {
            int j = _occa_tiled_j - get_local_id(1);
            if (j > 0) {
              ab[i] = add(a[i], b[j - 1]);
            }
          }
        }
      }
    }
  }
}

// Outer -> outer ==> inner -> regular
__kernel void _occa_addVectors4_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors4_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 1) * get_group_id(0));
    {
      int i = _occa_tiled_i - ((1) * get_group_id(1));
      if (i >= 0) {
        {
          int _occa_tiled_j = (entries) - ((4) * get_local_id(1));
          for (int j = _occa_tiled_j; j > (_occa_tiled_j - (4)); --j) {
            if (j > 0) {
              ab[i] = add(a[i], b[j - 1]);
            }
          }
        }
      }
    }
  }
}

// Outer -> outer ==> inner -> inner
__kernel void _occa_addVectors5_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors5_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 1) * get_group_id(0));
    {
      int i = _occa_tiled_i - ((1) * get_group_id(1));
      if (i >= 0) {
        {
          int _occa_tiled_j = (entries) - ((4) * get_local_id(1));
          {
            int j = _occa_tiled_j - get_local_id(2);
            if (j > 0) {
              ab[i] = add(a[i], b[j - 1]);
            }
          }
        }
      }
    }
  }
}

// Outer -> outer ==> outer -> inner
__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors6_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors6_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 1) * get_group_id(0));
    {
      int i = _occa_tiled_i - ((1) * get_group_id(1));
      if (i >= 0) {
        {
          int _occa_tiled_j = (entries) - ((4) * get_group_id(2));
          {
            int j = _occa_tiled_j - get_local_id(0);
            if (j > 0) {
              ab[i] = add(a[i], b[j - 1]);
            }
          }
        }
      }
    }
  }
}
