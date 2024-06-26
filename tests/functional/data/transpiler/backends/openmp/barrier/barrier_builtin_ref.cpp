static float add(const float* a, int i, const float* b, int j) {
    return a[i] + b[j];
}

extern "C" void addVectors0(const int& N, const float* a, const float* b, float* ab) {
#pragma omp parallel for
    for (int i = 0; i < N; i += 4) {
        float s_b[4];
        const float* g_a = a;
        for (int j = 0; j < 4; ++j) {
            s_b[j] = b[i + j];
            ab[i + j] = add(g_a, i + j, s_b, j);
        }
    }
}
