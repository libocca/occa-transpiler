#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__constant int offset = 1;
// template<typename T>
float add(float a, float b);

float add(float a, float b) { return a + b + offset; }

// Outer -> inner
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
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner non 1 increment
__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors1_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors1_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 2) * get_group_id(0));
    {
      int i = _occa_tiled_i - ((2) * get_local_id(0));
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner unary post add
__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors2_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors2_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - ((4) * get_group_id(0));
    {
      int i = _occa_tiled_i - get_local_id(0);
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner unary pre add
__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors3_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors3_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - ((4) * get_group_id(0));
    {
      int i = _occa_tiled_i - get_local_id(0);
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner, check=True
__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors4_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors4_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 1) * get_group_id(0));
    {
      int i = _occa_tiled_i - ((1) * get_local_id(0));
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner, complex range
__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors5_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors5_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i =
        ((entries + 16)) - (((4) * (entries / 16 + 1)) * get_group_id(0));
    {
      int i = _occa_tiled_i - (((entries / 16 + 1)) * get_local_id(0));
      if (i >= (entries - 12 + 4)) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner, set dimension
__kernel __attribute__((reqd_work_group_size(1, 1, 4))) void
_occa_addVectors6_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(1, 1, 4))) void
_occa_addVectors6_0(const int entries, __global const float *a,
                    __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 1) * get_group_id(1));
    {
      int i = _occa_tiled_i - ((1) * get_local_id(2));
      if (i >= 0) {
        ab[i] = add(a[i], b[i]);
      }
    }
  }
}

// Outer -> inner ==> inner -> inner (nested)
__kernel void _occa_addVectors7_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors7_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i = (entries - 1) - (((4) * 1) * get_group_id(0));
    {
      int i = _occa_tiled_i - ((1) * get_local_id(0));
      if (i >= 0) {
        {
          int _occa_tiled_j = (entries - 1) - (((4) * 1) * get_local_id(1));
          {
            int j = _occa_tiled_j - ((1) * get_local_id(2));
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
__kernel void _occa_addVectors8_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors8_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int _occa_tiled_i =
        ((entries + 16)) - (((4) * (entries / 16 + 1)) * get_group_id(0));
    {
      int i = _occa_tiled_i - (((entries / 16 + 1)) * get_local_id(0));
      if (i >= (entries - 12 + static_cast<int>(*a))) {
        {
          unsigned long long _occa_tiled_j =
              ((entries + 16)) - (((4) * (entries / 16 + 1)) * get_local_id(1));
          {
            unsigned long long j =
                _occa_tiled_j - (((entries / 16 + 1)) * get_local_id(2));
            if (j >= (entries - 12 + static_cast<int>(*a))) {
              ab[i] = add(a[i], b[j]);
            }
          }
        }
      }
    }
  }
}
