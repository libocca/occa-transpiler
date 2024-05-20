#pragma OPENCL EXTENSON cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(7, 5, 2))) void
_occa_test0_0(const int entries, __global const float *a,
              __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(7, 5, 2))) void
_occa_test0_0(const int entries, __global const float *a,
              __global const float *b, __global float *ab) {
  {
    int x = (0) + get_group_id(2);
    // int before1 = 1 + before0;
    int before1 = 1;
    {
      int y = (0) + get_group_id(1);
      int before2 = 1 + before1;
      {
        int z = (0) + get_group_id(0);
        int before3 = 1 + before2;
        {
          int n = (0) + get_local_id(2);
          int after0 = 1 + before3;
          {
            int m = (0) + get_local_id(1);
            int after1 = 1 + after0;
            {
              int k = (0) + get_local_id(0);
              int after2 = 1 + after1;
              ab[x] = a[x] + b[x] +
                      static_cast<float>(k + m + n + z + y + x + after2);
            }
            {
              int k = (0) + get_local_id(0);
              int after2 = 1 + after1;
              ab[x] = a[x] + b[x] +
                      static_cast<float>(k + m + n + z + y + x + after2);
            }
          }
          {
            int m = (0) + get_local_id(1);
            int after1 = 1 + after0;
            {
              int k = (0) + get_local_id(0);
              int after2 = 1 + after1;
              ab[x] = a[x] + b[x] +
                      static_cast<float>(k + m + n + z + y + x + after2);
            }
            {
              int k = (0) + get_local_id(0);
              int after2 = 1 + after1;
              ab[x] = a[x] + b[x] +
                      static_cast<float>(k + m + n + z + y + x + after2);
            }
          }
        }
      }
    }
  }
}

__kernel __attribute__((reqd_work_group_size(7, 5, 2))) void
_occa_test0_1(const int entries, __global const float *a,
              __global const float *b, __global float *ab);

__kernel __attribute__((reqd_work_group_size(7, 5, 2))) void
_occa_test0_1(const int entries, __global const float *a,
              __global const float *b, __global float *ab) {
  {
    int x = (0) + get_group_id(2);
    // int before1 = 1 + before00;
    int before1 = 1;
    {
      int y = (0) + get_group_id(1);
      int before2 = 1 + before1;
      {
        int z = (0) + get_group_id(0);
        int before3 = 1 + before2;
        {
          int n = (0) + get_local_id(2);
          int after0 = 1 + before3;
          {
            int m = (0) + get_local_id(1);
            int after1 = 1 + after0;
            {
              int k = (0) + get_local_id(0);
              int after2 = 1 + after1;
              ab[x] = a[x] + b[x] +
                      static_cast<float>(k + m + n + z + y + x + after2);
            }
            {
              int k = (0) + get_local_id(0);
              int after2 = 1 + after1;
              ab[x] = a[x] + b[x] +
                      static_cast<float>(k + m + n + z + y + x + after2);
            }
          }
          {
            int m = (0) + get_local_id(1);
            int after1 = 1 + after0;
            {
              int k = (0) + get_local_id(0);
              int after2 = 1 + after1;
              ab[x] = a[x] + b[x] +
                      static_cast<float>(k + m + n + z + y + x + after2);
            }
            {
              int k = (0) + get_local_id(0);
              int after2 = 1 + after1;
              ab[x] = a[x] + b[x] +
                      static_cast<float>(k + m + n + z + y + x + after2);
            }
          }
        }
      }
    }
  }
}
