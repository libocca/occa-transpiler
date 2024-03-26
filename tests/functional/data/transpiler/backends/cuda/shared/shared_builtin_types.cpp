@kernel void function1(const int* data) {
    @outer for (int i = 0; i < 64; ++i) {
        @shared int arr1[32];
        @shared float arr2[8][32];
        @shared double arr3[4 + 4];
        @inner for (int j = 0; j < 64; ++j) {
        }
    }
}
