static float add(const float* a, int i, const float* b, int j) {
    return a[i] + b[j];
}

extern "C" void addVectors0(const int& N, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int i = 0; i < N; i += 4) {
        int _occa_exclusive_index;
        float s_b[4];
        const float* g_a[4] = {a};
        _occa_exclusive_index = 0;
        for (int j = 0; j < 4; ++j) {
            s_b[j] = b[i + j];
            ab[i + j] = add(g_a[_occa_exclusive_index], i + j, s_b, j);
            ++_occa_exclusive_index;
        }
    }
}
