[[okl::dim("(4,4)")]] typedef float* mat4;

[[okl::kernel("")]] void addVectors(const int entries,
                                          [[okl::dim("(x,y)")]] const float* a,
                                          [[okl::dim("(y,x)")]] const float* b,
                                          float* ab) {
  [[okl::dimOrder("(1,0)")]] mat4 arr1 = ab;
  arr1(1, 1) = 0;
  [[okl::dim("(x,y)")]] int arr2[12];
  [[okl::dim("(x,y)")]] int arr3[12] = {0};
}
