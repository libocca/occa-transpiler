static float add(const float* a, int i, const float* b, int j) {
    return a[i] + b[j];
}

#define BLOCK_SIZE 4

extern "C" void addVectors0(const int& N, const float* a, const float* b, float* ab) {
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        int _occa_exclusive_index;
        float s_b[BLOCK_SIZE];
        const float* g_a[4] = {a};
        _occa_exclusive_index = 0;
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            s_b[j] = b[i + j];
            ;

            ab[i + j] = add(g_a[_occa_exclusive_index], i + j, s_b, j);
            ++_occa_exclusive_index;
        }
    }
}
