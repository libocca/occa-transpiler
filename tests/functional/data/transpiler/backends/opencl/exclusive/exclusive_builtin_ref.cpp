#pragma OPENCL EXTENSON cl_khr_fp64 : enable

static float add(const float *a, int i, const float *b, int j);

static float add(const float *a, int i, const float *b, int j) {
  return a[i] + b[j];
}

// TODO: fix preprocessor handling and try with define
// #define BLOCK_SIZE 4
__constant int BLOCK_SIZE = 4;
__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors_0(const int N, __global const float *a,
                   __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void
_occa_addVectors_0(const int N, __global const float *a,
                   __global const float *b, __global float *ab) {
  {
    int i = (0) + ((BLOCK_SIZE)*get_group_id(0));
    __local float s_b[BLOCK_SIZE];
    const float *g_a = a;
    {
      int j = (0) + get_local_id(0);
      s_b[j] = b[i + j];
      barrier(CLK_LOCAL_MEM_FENCE);
      ab[i + j] = add(g_a, i + j, s_b, j);
    }
  }
}
