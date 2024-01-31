typedef float* mat4 @dim(4, 4);

@kernel void addVectors(const int entries, @dim(x,y) const float *a, const float *b @dim(y,x), float *ab) {
  @tile(4, @outer, @inner)
    for (int i = 0; i < 4; ++i) {
    // Single
    {
      mat4 @dimOrder(1, 0) arr1 = ab;
      arr1(1, 1) = 0;
      int @dim(x,y) arr2[12];
      int arr3[12] @dim(x,y) = { 0 };
    };

    // Multiple
    {
      @dim(x,y) int arr1_1[12], arr1_2[12];
      int @dim(x,y) arr2_1[12], arr2_2[12];
      int arr3_1[12] @dim(x,y), arr3_2[12] @dim(y,x);
    };
  };
}
