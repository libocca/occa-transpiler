typedef [[okl::dim("(4,4)")]] float *mat4;

[[okl::kernel("(void)")]] void addVectors(const int entries,
                                          [[okl::dim("(x,y)")]] const float *a,
                                          [[okl::dim("(y,x)")]] const float *b,
                                          float *ab) {
  [[okl::tile("(4,@outer,@inner)")]] for (int i = 0; i < 4; ++i) {
    // Single
    {
      [[okl::dimOrder("(1,0)")]] mat4 arr1 = ab;
      arr1(1, 1) = 0;
      [[okl::dim("(x,y)")]] int arr2[12];
      [[okl::dim("(x,y)")]] int arr3[12] = {0};
    };

    // Multiple
    {
      [[okl::dim("(x,y)")]] int arr1_1[12], arr1_2[12];
      int [[okl::dim("(x,y)")]] arr2_1[12], arr2_2[12];
      int [[okl::dim("(x,y)")]] arr3_1[12], [[okl::dim("(y,x)")]] arr3_2[12];
    };
  };
}
