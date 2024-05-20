#pragma OPENCL EXTENSON cl_khr_fp64 : enable

constexpr float f = 13;

class HelloClass {
public:
  static constexpr int a = 2 + 2;
};

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void _occa_test_0();

__kernel __attribute__((reqd_work_group_size(10, 1, 1))) void _occa_test_0() {
  {
    int i = (0) + get_group_id(0);
    { int j = (0) + get_local_id(0); }
  }
}
