@kernel void addVectors(const int entries, const float* a, const float* b, float* ab) {
    @dim(x1, y1) int arr1_1[12], arr1_2[12];
    int @dim(x2, y2) arr2_1[12], arr2_2[12];
    int arr3_1[12] @dim(x3, y3), arr3_2[12] @dim(y4, x4);
}
