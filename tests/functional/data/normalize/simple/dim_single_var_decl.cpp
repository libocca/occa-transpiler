typedef float* mat4 @dim(4, 4);

@kernel void addVectors(const int entries,
                        @dim(x, y) const float* a,
                        const float* b @dim(y, x),
                        float* ab) {
    mat4 @dimOrder(1, 0) arr1 = ab;
    arr1(1, 1) = 0;
    int @dim(x, y) arr2[12];
    int arr3[12] @dim(x, y) = {0};
}
