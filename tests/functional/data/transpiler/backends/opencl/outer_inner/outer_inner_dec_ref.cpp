#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__constant int offset = 1;
// template<typename T>
float add(float a, float b);

float add(float a, float b) { return a + b + offset; }

// Outer -> inner
__kernel void _occa_addVectors0_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors0_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int j = (entries - 1) - ((1) * get_group_id(0));
    {
      int i = (entries - 1) - ((1) * get_local_id(0));
      ab[i] = add(a[i], b[i]);
    }
  }
}

// Outer -> inner non 1 increment
__kernel void _occa_addVectors1_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors1_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int j = (entries - 1) - ((2) * get_group_id(0));
    {
      int i = (entries - 1) - ((2) * get_local_id(0));
      ab[i] = add(a[i], b[i]);
    }
  }
}

// Outer -> inner unary post add
__kernel void _occa_addVectors2_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors2_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int j = (entries - 1) - get_group_id(0);
    {
      int i = (entries)-get_local_id(0);
      ab[i] = add(a[i], b[i]);
    }
  }
}

// Outer -> inner unary pre add
__kernel void _occa_addVectors3_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors3_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int j = (entries - 1) - get_group_id(0);
    {
      int i = (entries - 1) - get_local_id(0);
      ab[i] = add(a[i], b[i]);
    }
  }
}

// Outer -> outer -> inner -> inner
// TODO: change after sema calculates dimensions
__kernel void _occa_addVectors4_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors4_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int i = (entries - 1) - get_group_id(1);
    {
      int j = (entries - 1) - get_group_id(0);
      {
        int k = (entries - 1) - get_local_id(1);
        {
          int ii = (entries - 1) - get_local_id(0);
          ab[ii + k] = add(a[i], b[j]);
        }
      }
    }
  }
}

// Outer -> outer -> inner -> inner + manual dimensions specification
__kernel void _occa_addVectors5_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors5_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int i = (entries - 1) - get_group_id(1);
    {
      int j = (entries - 1) - get_group_id(0);
      {
        int k = (entries - 1) - get_local_id(1);
        {
          int ii = (entries - 1) - get_local_id(0);
          ab[ii + k] = add(a[i], b[j]);
        }
      }
    }
  }
}

// Outer -> outer -> inner -> inner + partially manual dimensions specification
// TODO: change after sema calculates dimensions
__kernel void _occa_addVectors6_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab);

__kernel void _occa_addVectors6_0(const int entries, __global const float *a,
                                  __global const float *b, __global float *ab) {
  {
    int i = (entries - 1) - get_group_id(1);
    {
      int j = (entries - 1) - get_group_id(0);
      {
        int k = (entries - 1) - get_local_id(1);
        {
          int ii = (entries - 1) - get_local_id(0);
          ab[ii + k] = add(a[i], b[j]);
        }
      }
    }
  }
}
