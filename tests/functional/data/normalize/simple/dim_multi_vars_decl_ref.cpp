[[okl::kernel("")]] void addVectors(const int entries,
                                          const float* a,
                                          const float* b,
                                          float* ab) {
  [[okl::dim("(x1,y1)")]] int arr1_1[12], arr1_2[12];
  [[okl::dim("(x2,y2)")]] int arr2_1[12], arr2_2[12];
  [[okl::dim("(x3,y3)")]] [[okl::dim("(y4,x4)")]] int arr3_1[12], arr3_2[12];
}
