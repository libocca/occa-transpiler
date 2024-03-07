extern "C" void function1(const int* data) {
#pragma omp parallel for
    for (int i = 0; i < 32; ++i) {
        int arr1[32];
        float arr2[8][32];
        double arr3[4 + 4];
        for (int j = 0; j < 8; ++j) {
            arr1[i] += int(arr2[j][i] * arr3[j]);
        }
    }
}
