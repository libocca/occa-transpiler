typedef float ex_float32_t;

extern "C" void test_kernel() {
#pragma omp parallel for
    for (int i = 0; i < 32; ++i) {
        int _occa_exclusive_index;
        ex_float32_t d[32][32];
        _occa_exclusive_index = 0;
        for (int j = 0; j < 32; ++j) {
            d[_occa_exclusive_index][j] = i - j;
            ++_occa_exclusive_index;
        }
    }
}
