#pragma OPENCL EXTENSON cl_khr_fp64 : enable

static float add1(const float *a, int i, const float *b, int j);

static float add1(const float *a, int i, const float *b, int j) {
  return a[i] + b[i];
}

float add2(const float *a, int i, const float *b, int j);

float add2(const float *a, int i, const float *b, int j) { return a[i] + b[i]; }

// At least one @kern function is requried
__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void _occa_kern_0();

__kernel __attribute__((reqd_work_group_size(32, 1, 1))) void _occa_kern_0() {
  {
    int i = (0) + get_group_id(0);
    { int j = (0) + get_local_id(0); }
  }
}
