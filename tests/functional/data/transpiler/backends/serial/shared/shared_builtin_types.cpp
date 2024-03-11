@kernel void function1(const int* data) {
    for (int i = 0; i < 32; ++i; @outer) {
        @shared int arr1[32];
        @shared float arr2[8][32];
        @shared double arr3[4 + 4];
        for (int j = 0; j < 8; ++j; @inner) {
            arr1[i] += int(arr2[j][i] * arr3[j]);
        }
    }
}
